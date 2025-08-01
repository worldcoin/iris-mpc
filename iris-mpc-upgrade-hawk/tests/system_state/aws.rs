use crate::utils::{errors::TestError, logger};
use aws_sdk_s3::{primitives::ByteStream as S3_ByteStream, Client as S3_Client};
use eyre::Result;
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_cpu::genesis::utils::aws::{
    get_s3_bucket_for_iris_deletions, get_s3_key_for_iris_deletions, IrisDeletionsForS3,
};

/// Component name for logging purposes.
const COMPONENT: &str = "SystemState-Aws";

/// Inserts serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `s3_client` - A configured AWS S3 client instance.
/// * `deletions` - Iris serial identifiers to be marked as deleted.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
pub async fn insert_iris_deletions(
    config: &NodeConfig,
    s3_client: &S3_Client,
    deletions: Vec<IrisSerialId>,
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
            deleted_serial_ids: deletions,
        })
        .unwrap()
        .into_bytes(),
    );

    // Upload payload.
    s3_client
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
