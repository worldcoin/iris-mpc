use crate::utils::{errors::TestError, logger};
use aws_config;
use aws_sdk_s3::{
    config::{Builder as S3_ConfigBuilder, Region as AWS_Region},
    primitives::ByteStream as S3_ByteStream,
    Client as S3_Client,
};
use eyre::Result;
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig, ENV_PROD, ENV_STAGE},
    IrisSerialId,
};
use iris_mpc_cpu::genesis::utils::aws::{
    get_s3_bucket_for_iris_deletions, get_s3_key_for_iris_deletions, IrisDeletionsForS3,
};

/// Component name for logging purposes.
const COMPONENT: &str = "SystemState-Aws";

/// Default AWS region.  TODO: remove.
const DEFAULT_AWS_REGION: &str = "eu-north-1";

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

/// Returns an S3 client with retry configuration.
async fn get_s3_client(config: &NodeConfig) -> Result<S3_Client> {
    let region = config
        .to_owned()
        .aws
        .and_then(|aws| aws.region)
        .unwrap_or_else(|| DEFAULT_AWS_REGION.to_owned());
    let region_provider = AWS_Region::new(region);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
    let retry_config = aws_config::retry::RetryConfig::standard().with_max_attempts(5);
    let s3_config = S3_ConfigBuilder::from(&shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    Ok(S3_Client::from_conf(s3_config))
}
