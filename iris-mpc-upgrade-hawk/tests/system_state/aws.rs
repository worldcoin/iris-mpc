use crate::utils::{aws::get_s3_client, errors::TestError, logger, types::NetConfig};
use aws_sdk_s3::primitives::ByteStream as S3_ByteStream;
use eyre::Result;
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_cpu::genesis::utils::aws::{
    get_s3_bucket_for_iris_deletions, get_s3_key_for_iris_deletions, IrisDeletionsForS3,
};

/// Component name for logging purposes.
const COMPONENT: &str = "STATE-AWS";

/// Uploads a set of serial identifiers marked as deleted to each node's AWS S3 bucket.
///
/// # Arguments
///
/// * `net_config` - Network wide configuration.
/// * `data` - Iris serial identifiers to be marked as deleted.
///
pub async fn upload_iris_deletions(
    net_config: &NetConfig,
    data: &Vec<IrisSerialId>,
) -> Result<(), TestError> {
    for node_config in net_config.iter() {
        upload_iris_deletions_node(node_config, data).await.unwrap();
    }

    Ok(())
}

/// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
async fn upload_iris_deletions_node(
    config: &NodeConfig,
    data: &Vec<IrisSerialId>,
) -> Result<(), TestError> {
    // Set bucket/key based on environment.
    let s3_bucket = get_s3_bucket_for_iris_deletions(config);
    let s3_key = get_s3_key_for_iris_deletions(config);
    logger::log_info(
        COMPONENT,
        format!(
            "Inserting deleted serial ids into S3 bucket: {}, key: {}",
            s3_bucket, s3_key
        )
        .as_str(),
    );

    // Set body of payload to be persisted.
    let body = S3_ByteStream::from(
        serde_json::to_string(&IrisDeletionsForS3 {
            deleted_serial_ids: data.to_owned(),
        })
        .unwrap()
        .into_bytes(),
    );

    // Upload payload.
    get_s3_client(config)
        .await
        .unwrap()
        .put_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .body(body)
        .send()
        .await
        .map_err(|err| {
            logger::log_error(COMPONENT, format!("Failed to upload file to S3: {}", err));
            TestError::SetupError("Failed to upload Iris deletions to S3".to_string())
        })?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::{upload_iris_deletions, NetConfig};
    use crate::{
        resources::{self, NODE_CONFIG_KIND_GENESIS},
        utils::aws::get_s3_client,
    };

    fn get_net_config() -> NetConfig {
        resources::read_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap()
    }

    #[tokio::test]
    async fn test_get_s3_client_and_list_buckets() {
        for cfg in get_net_config() {
            let n_buckets = get_s3_client(&cfg)
                .await
                .unwrap()
                .list_buckets()
                .send()
                .await
                .unwrap()
                .buckets
                .unwrap()
                .iter()
                .len();
            assert!(n_buckets > 0);
        }
    }

    #[tokio::test]
    async fn test_upload_iris_deletions() {
        let cfg = get_net_config();

        match upload_iris_deletions(&cfg, &vec![]).await {
            Ok(_) => (),
            Err(err) => panic!("Failed to upload Iris deletions: Err={}", err),
        };
    }

    #[tokio::test]
    async fn test_upload_iris_deletions_node() {
        println!("TODO");
    }
}
