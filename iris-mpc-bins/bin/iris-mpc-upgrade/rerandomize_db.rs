use std::ops::Range;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use clap::Parser;
use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::galois;
use iris_mpc_common::galois::degree4::basis::Monomial;
use iris_mpc_common::galois::degree4::GaloisRingElement;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::id::PartyID;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{DbStoredIris, Store, StoredIrisRef};
use iris_mpc_upgrade::config::{
    KeyCleanupConfig, KeyGenConfig, ReRandomizeCheckConfig, ReRandomizeConfig,
    ReRandomizeDbSubCommand,
};
use iris_mpc_upgrade::rerandomization::randomize_iris;
use iris_mpc_upgrade::tripartite_dh;
use iris_mpc_upgrade::{
    config::ReRandomizeDbConfig,
    utils::{install_tracing, spawn_healthcheck_server},
};
use itertools::Itertools;
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
        ReRandomizeDbSubCommand::RerandomizeCheck(config) => rerandomize_check_main(config).await,
    }
}

const PUBLIC_KEY_S3_KEY_NAME_PREFIX: &str = "iris-mpc-tripartite-ecdh-public-key-party";

async fn keygen_main(config: KeyGenConfig) -> Result<()> {
    let sdk_config = aws_config::from_env().load().await;

    let bucket_key_name = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, config.party_id);
    let private_key_secret_id: String = format!(
        "{}/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-{}",
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
        "{}/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-{}",
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
        "{}/iris-mpc-db-rerandomization/tripartite-ecdh-private-key-{}",
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

async fn rerandomize_check_main(config: ReRandomizeCheckConfig) -> Result<()> {
    tracing::info!("Starting rerandomize check across old and new databases");

    let store_old_0 =
        build_read_only_store(&config.old_db_url_party_0, &config.old_schema_name_party_0).await?;
    let store_old_1 =
        build_read_only_store(&config.old_db_url_party_1, &config.old_schema_name_party_1).await?;
    let store_old_2 =
        build_read_only_store(&config.old_db_url_party_2, &config.old_schema_name_party_2).await?;

    let store_new_0 =
        build_read_only_store(&config.new_db_url_party_0, &config.new_schema_name_party_0).await?;
    let store_new_1 =
        build_read_only_store(&config.new_db_url_party_1, &config.new_schema_name_party_1).await?;
    let store_new_2 =
        build_read_only_store(&config.new_db_url_party_2, &config.new_schema_name_party_2).await?;

    let max_id = store_old_0.get_max_serial_id().await?;
    for store in [
        &store_old_1,
        &store_old_2,
        &store_new_0,
        &store_new_1,
        &store_new_2,
    ] {
        let store_max_id = store.get_max_serial_id().await?;
        if store_max_id != max_id {
            eyre::bail!("Mismatched max serial IDs between databases.");
        }
    }

    let end = max_id as u64 + 1;

    let mut stream_old_0 = store_old_0.stream_irises_in_range(1..end);
    let mut stream_old_1 = store_old_1.stream_irises_in_range(1..end);
    let mut stream_old_2 = store_old_2.stream_irises_in_range(1..end);
    let mut stream_new_0 = store_new_0.stream_irises_in_range(1..end);
    let mut stream_new_1 = store_new_1.stream_irises_in_range(1..end);
    let mut stream_new_2 = store_new_2.stream_irises_in_range(1..end);

    loop {
        let iris_old_0 = stream_old_0.try_next().await?;
        let iris_old_1 = stream_old_1.try_next().await?;
        let iris_old_2 = stream_old_2.try_next().await?;
        let iris_new_0 = stream_new_0.try_next().await?;
        let iris_new_1 = stream_new_1.try_next().await?;
        let iris_new_2 = stream_new_2.try_next().await?;

        if [
            &iris_old_0,
            &iris_old_1,
            &iris_old_2,
            &iris_new_0,
            &iris_new_1,
            &iris_new_2,
        ]
        .iter()
        .all(|entry| entry.is_none())
        {
            break;
        }

        let iris_old_0 = iris_old_0
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;
        let iris_old_1 = iris_old_1
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;
        let iris_old_2 = iris_old_2
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;
        let iris_new_0 = iris_new_0
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;
        let iris_new_1 = iris_new_1
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;
        let iris_new_2 = iris_new_2
            .ok_or_else(|| eyre::eyre!("Mismatched number of irises between databases"))?;

        if iris_new_0.id() != iris_old_0.id()
            || iris_new_1.id() != iris_old_1.id()
            || iris_new_2.id() != iris_old_2.id()
            || iris_old_0.id() != iris_old_1.id()
            || iris_old_0.id() != iris_old_2.id()
        {
            eyre::bail!(
                "Mismatched iris IDs between databases: {}, {}, {}, {}, {}, {}",
                iris_old_0.id(),
                iris_old_1.id(),
                iris_old_2.id(),
                iris_new_0.id(),
                iris_new_1.id(),
                iris_new_2.id(),
            );
        }
        let id = iris_old_0.id();

        for (iris_new, iris_old) in [
            (&iris_new_0, &iris_old_0),
            (&iris_new_1, &iris_old_1),
            (&iris_new_2, &iris_old_2),
        ] {
            if iris_new.left_code() == iris_old.left_code() {
                eyre::bail!("Left code not changed for iris ID {id} between databases");
            }
            if iris_new.left_mask() == iris_old.left_mask() {
                eyre::bail!("Left mask not changed for iris ID {id} between databases");
            }
            if iris_new.right_code() == iris_old.right_code() {
                eyre::bail!("Right code not changed for iris ID {id} between databases");
            }
            if iris_new.right_mask() == iris_old.right_mask() {
                eyre::bail!("Right mask not changed for iris ID {id} between databases");
            }
        }

        let (iris_0, iris_1, iris_2) = [iris_old_0, iris_old_1, iris_old_2]
            .iter()
            .enumerate()
            .map(|(party_id, iris)| {
                (
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_mask().try_into().unwrap(),
                    },
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_mask().try_into().unwrap(),
                    },
                )
            })
            .collect_tuple()
            .unwrap();
        let old_left_code = reconstruct_shares(&iris_0.0.coefs, &iris_1.0.coefs, &iris_2.0.coefs);
        let old_left_mask = reconstruct_shares(&iris_0.1.coefs, &iris_1.1.coefs, &iris_2.1.coefs);
        let old_right_code = reconstruct_shares(&iris_0.2.coefs, &iris_1.2.coefs, &iris_2.2.coefs);
        let old_right_mask = reconstruct_shares(&iris_0.3.coefs, &iris_1.3.coefs, &iris_2.3.coefs);

        let (iris_3, iris_4, iris_5) = [iris_new_0, iris_new_1, iris_new_2]
            .iter()
            .enumerate()
            .map(|(party_id, iris)| {
                (
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.left_mask().try_into().unwrap(),
                    },
                    GaloisRingIrisCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_code().try_into().unwrap(),
                    },
                    GaloisRingTrimmedMaskCodeShare {
                        id: party_id + 1,
                        coefs: iris.right_mask().try_into().unwrap(),
                    },
                )
            })
            .collect_tuple()
            .unwrap();
        let new_left_code = reconstruct_shares(&iris_3.0.coefs, &iris_4.0.coefs, &iris_5.0.coefs);
        let new_left_mask = reconstruct_shares(&iris_3.1.coefs, &iris_4.1.coefs, &iris_5.1.coefs);
        let new_right_code = reconstruct_shares(&iris_3.2.coefs, &iris_4.2.coefs, &iris_5.2.coefs);
        let new_right_mask = reconstruct_shares(&iris_3.3.coefs, &iris_4.3.coefs, &iris_5.3.coefs);

        if old_left_code != new_left_code {
            eyre::bail!("Mismatched left code for iris ID {id} between databases");
        }
        if old_left_mask != new_left_mask {
            eyre::bail!("Mismatched left mask for iris ID {id} between databases");
        }
        if old_right_code != new_right_code {
            eyre::bail!("Mismatched right code for iris ID {id} between databases");
        }
        if old_right_mask != new_right_mask {
            eyre::bail!("Mismatched right mask for iris ID {id} between databases");
        }
    }

    tracing::info!("All irises match between databases.");

    Ok(())
}

async fn download_public_key(config: &ReRandomizeConfig, party_id: u8) -> Result<String> {
    if config.env == "testing" {
        let bucket = config.public_key_bucket_name.as_ref().ok_or_else(|| {
            eyre::eyre!("PUBLIC_KEY_BUCKET_NAME must be provided when ENVIRONMENT=testing")
        })?;
        if bucket.trim().is_empty() {
            return Err(eyre::eyre!(
                "PUBLIC_KEY_BUCKET_NAME must not be empty when ENVIRONMENT=testing"
            ));
        }
        download_public_key_from_localstack(bucket, party_id).await
    } else {
        let base_url = config.public_key_base_url.as_ref().ok_or_else(|| {
            eyre::eyre!("PUBLIC_KEY_BASE_URL must be provided when ENVIRONMENT is not testing")
        })?;
        if base_url.trim().is_empty() {
            return Err(eyre::eyre!(
                "PUBLIC_KEY_BASE_URL must not be empty when ENVIRONMENT is not testing"
            ));
        }
        download_public_key_from_http(base_url, party_id).await
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

async fn build_read_only_store(db_url: &str, schema_name: &str) -> Result<Store> {
    let postgres_client = PostgresClient::new(db_url, schema_name, AccessMode::ReadOnly).await?;
    Store::new(&postgres_client).await
}

fn reconstruct_shares(share0: &[u16], share1: &[u16], share2: &[u16]) -> Vec<u16> {
    let lag_01 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID0,
        PartyID::ID1,
    );
    let lag_10 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID1,
        PartyID::ID0,
    );
    let lag_02 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID0,
        PartyID::ID2,
    );
    let lag_20 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID2,
        PartyID::ID0,
    );
    let lag_12 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID1,
        PartyID::ID2,
    );
    let lag_21 = galois::degree4::ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(
        PartyID::ID2,
        PartyID::ID1,
    );

    assert!(share0.len() == share1.len() && share1.len() == share2.len());

    let recon01 = share0
        .chunks_exact(4)
        .zip_eq(share1.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_01 + b * lag_10;
            c.coefs
        })
        .collect_vec();
    let recon12 = share1
        .chunks_exact(4)
        .zip_eq(share2.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_12 + b * lag_21;
            c.coefs
        })
        .collect_vec();
    let recon02 = share0
        .chunks_exact(4)
        .zip_eq(share2.chunks_exact(4))
        .flat_map(|(a, b)| {
            let a = GaloisRingElement::<Monomial>::from_coefs(a.try_into().unwrap());
            let b = GaloisRingElement::<Monomial>::from_coefs(b.try_into().unwrap());
            let c = a * lag_02 + b * lag_20;
            c.coefs
        })
        .collect_vec();

    assert_eq!(recon01, recon12);
    assert_eq!(recon01, recon02);
    recon01
}

async fn download_public_key_from_localstack(bucket: &str, party_id: u8) -> Result<String> {
    let key = format!("{}-{}", PUBLIC_KEY_S3_KEY_NAME_PREFIX, party_id);
    let request_url = format!("http://localhost:4566/{}/{}", bucket, key);
    tracing::info!("Downloading public key from localstack {}", request_url);
    let client = reqwest::Client::new();
    let response = client.get(&request_url).send().await?.text().await?;
    Ok(response)
}
