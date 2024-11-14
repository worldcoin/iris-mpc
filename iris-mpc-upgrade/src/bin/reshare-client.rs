use clap::Parser;
use futures::StreamExt;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::ReShareClientConfig,
    proto::iris_mpc_reshare::{
        iris_code_re_share_service_client::IrisCodeReShareServiceClient, IrisCodeReShareStatus,
    },
    reshare::IrisCodeReshareSenderHelper,
};

const APP_NAME: &str = "SMPC";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let config = ReShareClientConfig::parse();

    // TODO: derive a common seed for the two participating parties
    let common_seed = [0u8; 32];

    let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
    let store = Store::new(&config.db_url, &schema_name).await?;

    let iris_stream = store.stream_irises_in_range(config.db_start..config.db_end);
    let mut iris_stream_chunks = iris_stream.chunks(config.batch_size as usize);

    let mut iris_reshare_helper = IrisCodeReshareSenderHelper::new(
        config.party_id as usize,
        config.other_party_id as usize,
        config.target_party_id as usize,
        common_seed,
    );

    let mut grpc_client = IrisCodeReShareServiceClient::connect(config.server_url).await?;

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
                    id:    config.party_id as usize + 1,
                    coefs: iris_code.left_code().try_into().unwrap(),
                },
                GaloisRingTrimmedMaskCodeShare {
                    id:    config.party_id as usize + 1,
                    coefs: iris_code.left_mask().try_into().unwrap(),
                },
                GaloisRingIrisCodeShare {
                    id:    config.party_id as usize + 1,
                    coefs: iris_code.right_code().try_into().unwrap(),
                },
                GaloisRingTrimmedMaskCodeShare {
                    id:    config.party_id as usize + 1,
                    coefs: iris_code.right_mask().try_into().unwrap(),
                },
            );
        }

        let request = iris_reshare_helper.finalize_reshare_batch();
        let mut timeout = tokio::time::Duration::from_millis(100);
        loop {
            let resp = grpc_client.re_share(request.clone()).await?;
            let resp = resp.into_inner();
            match resp.status {
                x if x == IrisCodeReShareStatus::Ok as i32 => {
                    break;
                }
                x if x == IrisCodeReShareStatus::FullQueue as i32 => {
                    tokio::time::sleep(timeout).await;
                    // todo: linear backoff strategy ok?
                    timeout += tokio::time::Duration::from_millis(100);
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
