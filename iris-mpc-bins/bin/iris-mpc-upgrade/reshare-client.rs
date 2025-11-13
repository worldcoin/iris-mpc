use clap::Parser;
use eyre::Result;
use futures::StreamExt;
use hkdf::Hkdf;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::kms_dh::derive_shared_secret,
};
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::ReShareClientConfig,
    proto::{
        get_size_of_reshare_iris_code_share_batch,
        iris_mpc_reshare::{
            iris_code_re_share_service_client::IrisCodeReShareServiceClient, IrisCodeReShareStatus,
        },
    },
    reshare::IrisCodeReshareSenderHelper,
    utils::{extract_domain, install_tracing},
};
use sha2::Sha256;
use tonic::transport::{Certificate, Channel, ClientTlsConfig};

const APP_NAME: &str = "SMPC";

async fn derive_common_seed(config: &ReShareClientConfig) -> Result<[u8; 32]> {
    let shared_secret = if config.environment == "testing" {
        // TODO: remove once localstack fixes KMS bug that returns different shared
        // secrets
        [0u8; 32]
    } else {
        derive_shared_secret(&config.my_kms_key_arn, &config.other_kms_key_arn).await?
    };

    let hk = Hkdf::<Sha256>::new(
        // sesstion id is used as salt
        Some(config.reshare_run_session_id.as_bytes()),
        &shared_secret,
    );
    let mut common_seed = [0u8; 32];
    // expand the common seed bound to the context "ReShare-Protocol-Client"
    hk.expand(b"ReShare-Protocol-Client", &mut common_seed)
        .map_err(|e| eyre::eyre!("error during HKDF expansion: {}", e))?;
    Ok(common_seed)
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let config = ReShareClientConfig::parse();

    let common_seed = derive_common_seed(&config).await?;

    let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
    let postgres_client =
        PostgresClient::new(&config.db_url, &schema_name, AccessMode::ReadWrite).await?;
    let store = Store::new(&postgres_client).await?;

    let iris_stream = store.stream_irises_in_range(config.db_start..config.db_end);
    let mut iris_stream_chunks = iris_stream.chunks(config.batch_size as usize);

    let mut iris_reshare_helper = IrisCodeReshareSenderHelper::new(
        config.party_id as usize,
        config.other_party_id as usize,
        config.target_party_id as usize,
        common_seed,
    );

    let encoded_message_size =
        get_size_of_reshare_iris_code_share_batch(config.batch_size as usize);
    if encoded_message_size > 100 * 1024 * 1024 {
        tracing::warn!(
            "encoded batch message size is large: {}MB",
            encoded_message_size as f64 / 1024.0 / 1024.0
        );
    }
    let encoded_message_size_with_buf = (encoded_message_size as f64 * 1.1) as usize;

    let pem = tokio::fs::read(config.ca_root_file_path)
        .await
        .expect("oh no, the cert file wasn't loaded");
    let cert = Certificate::from_pem(pem.clone());

    let domain = extract_domain(&config.server_url.clone(), true)?;
    println!(
        "TLS connecting to address {} using domain {},",
        config.server_url.clone(),
        domain
    );

    let tls = ClientTlsConfig::new()
        .domain_name(domain)
        .ca_certificate(cert);

    // build a tonic transport channel ourselves, since we want to add a tls config
    let channel = Channel::from_shared(config.server_url.clone())?
        .tls_config(tls)?
        .connect()
        .await?;

    let mut grpc_client = IrisCodeReShareServiceClient::new(channel)
        .max_decoding_message_size(encoded_message_size_with_buf)
        .max_encoding_message_size(encoded_message_size_with_buf);

    while let Some(chunk) = iris_stream_chunks.next().await {
        let iris_codes = chunk.into_iter().collect::<Result<Vec<_>, sqlx::Error>>()?;
        if iris_codes.is_empty() {
            continue;
        }
        let db_chunk_start = iris_codes.first().unwrap().id();
        let db_chunk_end = iris_codes.last().unwrap().id();

        // sanity check
        for window in iris_codes.as_slice().windows(2) {
            assert_eq!(
                window[0].id() + 1,
                window[1].id(),
                "expect consecutive iris codes"
            );
        }

        iris_reshare_helper.start_reshare_batch(db_chunk_start, db_chunk_end + 1);

        for iris_code in iris_codes {
            iris_reshare_helper.add_reshare_iris_to_batch(
                iris_code.id(),
                GaloisRingIrisCodeShare {
                    id: config.party_id as usize + 1,
                    coefs: iris_code.left_code().try_into().unwrap(),
                },
                GaloisRingTrimmedMaskCodeShare {
                    id: config.party_id as usize + 1,
                    coefs: iris_code.left_mask().try_into().unwrap(),
                },
                GaloisRingIrisCodeShare {
                    id: config.party_id as usize + 1,
                    coefs: iris_code.right_code().try_into().unwrap(),
                },
                GaloisRingTrimmedMaskCodeShare {
                    id: config.party_id as usize + 1,
                    coefs: iris_code.right_mask().try_into().unwrap(),
                },
            );
        }
        tracing::info!(
            "Submitting reshare request for iris codes {} to {}",
            db_chunk_start,
            db_chunk_end
        );

        let request = iris_reshare_helper.finalize_reshare_batch();
        let mut timeout = tokio::time::Duration::from_millis(config.retry_backoff_millis);
        loop {
            let resp = grpc_client.re_share(request.clone()).await?;
            let resp = resp.into_inner();
            match resp.status {
                x if x == IrisCodeReShareStatus::Ok as i32 => {
                    break;
                }
                x if x == IrisCodeReShareStatus::FullQueue as i32 => {
                    tokio::time::sleep(timeout).await;
                    timeout += tokio::time::Duration::from_millis(config.retry_backoff_millis);
                    continue;
                }
                x if x == IrisCodeReShareStatus::Error as i32 => {
                    return Err(eyre::eyre!(
                        "error during reshare request submission: {}",
                        resp.message
                    ));
                }
                _ => {
                    return Err(eyre::eyre!("unexpected reshare status: {}", resp.status));
                }
            }
        }
    }

    Ok(())
}
