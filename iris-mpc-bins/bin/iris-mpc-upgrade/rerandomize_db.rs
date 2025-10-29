use std::ops::Range;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use clap::Parser;
use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{DbStoredIris, Store, StoredIrisRef};
use iris_mpc_upgrade::config::{
    KeyCleanupConfig, KeyGenConfig, ReRandomizeConfig, ReRandomizeDbSubCommand,
};
use iris_mpc_upgrade::rerandomization::randomize_iris;
use iris_mpc_upgrade::tripartite_dh;
use iris_mpc_upgrade::{
    config::ReRandomizeDbConfig,
    utils::{install_tracing, spawn_healthcheck_server},
};
use tokio::task::JoinSet;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let config = ReRandomizeDbConfig::parse();

    match config.command {
        ReRandomizeDbSubCommand::RerandomizeDb(config) => rerandomize_db_main(config).await,
        ReRandomizeDbSubCommand::KeyGen(config) => keygen_main(config).await,
        ReRandomizeDbSubCommand::KeyCleanup(config) => keycleanup_main(config).await,
    }
}

const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "iris-mpc-tripartite-ecdh-public-key-party";

async fn keygen_main(config: KeyGenConfig) -> Result<()> {
    let sdk_config = aws_config::from_env().load().await;

    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, config.party_id);
    let private_key_secret_id: String = format!(
        "{}/iris-mpc/tripartite-ecdh-private-key-{}",
        config.env, config.party_id
    );

    let mut rng = rand::thread_rng();

    let secret_key = tripartite_dh::PrivateKey::random(&mut rng);
    let public_key = secret_key.public_key();
    let secret_key_bytes = secret_key.serialize();
    let public_key_bytes = public_key.serialize();
    let secret_key_b64 = STANDARD.encode(&secret_key_bytes);
    let public_key_b64 = STANDARD.encode(&public_key_bytes);
    let s3_config_builder = aws_sdk_s3::config::Builder::from(&sdk_config);
    let sm_config_builder = aws_sdk_secretsmanager::config::Builder::from(&sdk_config);
    let s3_client = S3Client::from_conf(s3_config_builder.build());
    let sm_client = SecretsManagerClient::from_conf(sm_config_builder.build());

    s3_client
        .put_object()
        .bucket(config.public_key_bucket_name)
        .key(bucket_key_name)
        .body(public_key_b64.to_string().into_bytes().into())
        .send()
        .await?;

    sm_client
        .create_secret()
        .name(private_key_secret_id)
        .secret_string(secret_key_b64)
        .send()
        .await?;

    Ok(())
}

async fn keycleanup_main(config: KeyCleanupConfig) -> Result<()> {
    let private_key_secret_id: String = format!(
        "{}/iris-mpc/tripartite-ecdh-private-key-{}",
        config.env, config.party_id
    );
    let sdk_config = aws_config::from_env().load().await;
    let sm_config_builder = aws_sdk_secretsmanager::config::Builder::from(&sdk_config);
    let sm_client = SecretsManagerClient::from_conf(sm_config_builder.build());
    sm_client
        .delete_secret()
        .secret_id(private_key_secret_id)
        .force_delete_without_recovery(true)
        .send()
        .await?;
    Ok(())
}

async fn rerandomize_db_main(config: ReRandomizeConfig) -> Result<()> {
    // Downloading the private key from secrets manager
    let sdk_config = aws_config::from_env().load().await;
    let sm_config_builder = aws_sdk_secretsmanager::config::Builder::from(&sdk_config);
    let sm_client = SecretsManagerClient::from_conf(sm_config_builder.build());
    let private_key_secret_id: String = format!(
        "{}/iris-mpc/tripartite-ecdh-private-key-{}",
        config.env, config.party_id
    );

    let secret_key_b64 = sm_client
        .get_secret_value()
        .secret_id(private_key_secret_id)
        .send()
        .await?
        .secret_string
        .ok_or_else(|| eyre::eyre!("Secret key not found in SecretManager"))?;
    let secret_key_bytes = STANDARD.decode(secret_key_b64)?;
    let private_key = tripartite_dh::PrivateKey::deserialize(&secret_key_bytes)
        .map_err(|_| eyre::eyre!("Failed to parse secret key"))?;

    // Downloading the public keys based on environment
    let next_id = (config.party_id + 1) % 3;
    let public_key_next = download_public_key(&config, next_id).await?;
    let public_key_next =
        tripartite_dh::PublicKeys::deserialize(&STANDARD.decode(public_key_next)?)
            .map_err(|_| eyre::eyre!("Failed to parse public key for party {}", next_id))?;
    let prev_id = (config.party_id + 2) % 3;
    let public_key_prev = download_public_key(&config, prev_id).await?;
    let public_key_prev =
        tripartite_dh::PublicKeys::deserialize(&STANDARD.decode(public_key_prev)?)
            .map_err(|_| eyre::eyre!("Failed to parse public key for party {}", prev_id))?;

    tracing::info!("Successfully downloaded keys from configured source and Secrets Manager");
    let shared_secret = private_key.derive_shared_secret(&public_key_next, &public_key_prev);
    let shared_secret_hash = blake3::hash(&shared_secret);
    tracing::info!(
        "Successfully derived shared secret, with hash: {} (This is not the actual shared secret!)",
        shared_secret_hash.to_hex()
    );

    tracing::info!("Starting healthcheck server.");

    let mut background_tasks = TaskMonitor::new();
    let _health_check_abort = background_tasks
        .spawn(async move { spawn_healthcheck_server(config.healthcheck_port).await });
    background_tasks.check_tasks();
    tracing::info!(
        "Healthcheck server running on port {}.",
        config.healthcheck_port.clone()
    );

    let postgres_client_read = PostgresClient::new(
        &config.source_db_url,
        &config.source_schema_name,
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_write = PostgresClient::new(
        &config.dest_db_url,
        &config.dest_schema_name,
        AccessMode::ReadWrite,
    )
    .await?;
    let read_store = Store::new(&postgres_client_read).await?;
    let write_store = Store::new(&postgres_client_write).await?;

    rerandomize_db(&read_store, &write_store, config, shared_secret).await?;

    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn rerandomize_db(
    read_store: &Store,
    write_store: &Store,
    config: ReRandomizeConfig,
    master_secret: [u8; 32],
) -> Result<()> {
    tracing::info!("Rerandomizing database for party ID: {}", config.party_id);

    let max_id = read_store.get_max_serial_id().await?;

    let start = std::cmp::max(1, config.range_min);
    let end = std::cmp::min(max_id, config.range_max_inclusive) + 1;

    let total = end - start;

    let chunk_len = total.div_ceil(config.num_tasks);

    let mut tasks = JoinSet::new();

    for i in 0..config.num_tasks {
        let start = start + i * chunk_len;
        let end = std::cmp::min(start + chunk_len, end);
        let read_store = read_store.clone();
        let write_store = write_store.clone();
        let party_id = config.party_id;
        let master_seed = master_secret;

        tasks.spawn(async move {
            for chunk_start in (start..end).step_by(config.chunk_size) {
                let chunk_end = std::cmp::min(chunk_start + config.chunk_size, end);
                let span = tracing::span!(
                    Level::INFO,
                    "Processing chunk",
                    chunk_start,
                    chunk_end,
                    party_id
                );
                let _span = span.enter();
                let chunk: Result<Vec<DbStoredIris>, _> = read_store
                    .stream_irises_in_range(Range {
                        start: chunk_start as u64,
                        end: chunk_end as u64,
                    })
                    .try_collect()
                    .await;
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Failed to fetch chunk {chunk_start}-{chunk_end}: {e}");
                        continue;
                    }
                };

                tracing::info!(
                    "Fetched chunk [{},{}) for party ID: {}",
                    chunk_start,
                    chunk_end,
                    party_id
                );

                let rerandomized_chunk: Vec<_> = chunk
                    .into_iter()
                    .map(|iris| randomize_iris(iris, &master_seed, party_id as usize))
                    .collect();
                tracing::info!(
                    "Rerandomized chunk [{},{}) for party ID: {}",
                    chunk_start,
                    chunk_end,
                    party_id
                );

                let inserted_chunk: Vec<_> = rerandomized_chunk
                    .iter()
                    .map(
                        |(id, left_code, left_mask, right_code, right_mask)| StoredIrisRef {
                            id: *id,
                            left_code: &left_code.coefs,
                            left_mask: &left_mask.coefs,
                            right_code: &right_code.coefs,
                            right_mask: &right_mask.coefs,
                        },
                    )
                    .collect();

                let mut tx = match write_store.tx().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        tracing::error!("Failed to start transaction: {e}");
                        continue;
                    }
                };

                match write_store
                    .insert_irises_overriding(&mut tx, &inserted_chunk)
                    .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("Failed to insert rerandomized iris chunk: {e}");
                        continue;
                    }
                }

                if let Err(e) = tx.commit().await {
                    tracing::error!("Failed to commit transaction: {e}");
                } else {
                    tracing::info!(
                        "Successfully committed rerandomized chunk [{},{}) for party ID: {}",
                        chunk_start,
                        chunk_end,
                        party_id
                    );
                }
            }
        });
    }

    tasks.join_all().await;
    tracing::info!("All rerandomization tasks completed.");

    Ok(())
}

async fn download_public_key(config: &ReRandomizeConfig, party_id: u8) -> Result<String> {
    if config.env == "testing" {
        if config.public_key_bucket_name.trim().is_empty() {
            return Err(eyre::eyre!(
                "PUBLIC_KEY_BUCKET_NAME must be provided when ENVIRONMENT=testing"
            ));
        }
        download_public_key_from_localstack(&config.public_key_bucket_name, party_id).await
    } else {
        if config.public_key_base_url.trim().is_empty() {
            return Err(eyre::eyre!(
                "PUBLIC_KEY_BASE_URL must be provided when ENVIRONMENT is not testing"
            ));
        }
        download_public_key_from_http(&config.public_key_base_url, party_id).await
    }
}

async fn download_public_key_from_http(base_url: &str, party_id: u8) -> Result<String> {
    let normalized = base_url.trim_end_matches('/');
    let normalized = normalized.trim_end_matches('-');
    let request_url = format!("{}-{}", normalized, party_id);
    tracing::info!("Downloading public key from {}", request_url);
    let client = reqwest::Client::new();
    let response = client.get(&request_url).send().await?.text().await?;
    Ok(response)
}

async fn download_public_key_from_localstack(bucket: &str, party_id: u8) -> Result<String> {
    let key = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, party_id);
    let request_url = format!("http://localhost:4566/{}/{}", bucket, key);
    tracing::info!("Downloading public key from localstack {}", request_url);
    let client = reqwest::Client::new();
    let response = client.get(&request_url).send().await?.text().await?;
    Ok(response)
}
