use crate::utils::{errors::TestError, logger};
use aws_config;
use aws_sdk_s3::{
    config::{Builder as S3_ConfigBuilder, Region as AWS_Region},
    primitives::ByteStream as S3_ByteStream,
    Client as S3_Client,
};
use eyre::Result;
use iris_mpc_common::{
    config::{ENV_PROD, ENV_STAGE},
    IrisSerialId,
};
use serde::Serialize;

/// Component name for logging purposes.
const COMPONENT: &str = "SystemState-Aws";

/// Default AWS region.  TODO: remove.
const DEFAULT_AWS_REGION: &str = "eu-north-1";

/// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
pub async fn upload_iris_deletions(
    data: &Vec<IrisSerialId>,
    s3: &S3_Client,
    environment: &str,
) -> Result<(), TestError> {
    // Set bucket/key based on environment.
    let s3_bucket = get_s3_bucket_for_iris_deletions(environment);
    let s3_key = get_s3_key_for_iris_deletions(environment);
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
    s3.put_object()
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
pub async fn get_s3_client(region: Option<&str>, environment: &str) -> Result<S3_Client> {
    let region = region.unwrap_or(DEFAULT_AWS_REGION).to_string();
    let region_provider = AWS_Region::new(region);
    // TODO modify this to take all configuration values explicitly, instead of some from environment
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let force_path_style = environment != ENV_PROD && environment != ENV_STAGE;
    let retry_config = aws_config::retry::RetryConfig::standard().with_max_attempts(5);
    let s3_config = S3_ConfigBuilder::from(&shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    Ok(S3_Client::from_conf(s3_config))
}

// Struct for S3 serialization.
#[derive(Serialize, Debug, Clone)]
pub struct IrisDeletionsForS3 {
    pub deleted_serial_ids: Vec<IrisSerialId>,
}

/// AWS S3 bucket for iris deletions.
pub fn get_s3_bucket_for_iris_deletions(environment: &str) -> String {
    format!("wf-smpcv2-{}-sync-protocol", environment)
}

/// AWS S3 key for iris deletions.
pub fn get_s3_key_for_iris_deletions(environment: &str) -> String {
    format!("{}_deleted_serial_ids.json", environment)
}
