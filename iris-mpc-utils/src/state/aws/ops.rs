use crate::utils::logger;
use aws_sdk_s3::{primitives::ByteStream as S3_ByteStream, Client as S3_Client};
use eyre::{eyre, Result};
use iris_mpc::services::aws::clients::AwsClients;
use iris_mpc_common::{config::Config, IrisSerialId};
use serde::Serialize;

/// Component name for logging purposes.
const COMPONENT: &str = "State-Aws";

// Struct for S3 serialization.
#[derive(Serialize, Debug, Clone)]
pub struct IrisDeletionsForS3 {
    pub deleted_serial_ids: Vec<IrisSerialId>,
}

/// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
pub async fn upload_iris_deletions(
    data: &Vec<IrisSerialId>,
    s3: &S3_Client,
    environment: &str,
) -> Result<()> {
    /// AWS S3 bucket for iris deletions.
    fn get_s3_bucket_for_iris_deletions(environment: &str) -> String {
        format!("wf-smpcv2-{}-sync-protocol", environment)
    }

    /// AWS S3 key for iris deletions.
    fn get_s3_key_for_iris_deletions(environment: &str) -> String {
        format!("{}_deleted_serial_ids.json", environment)
    }

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
            eyre!("Failed to upload Iris deletions to S3")
        })?;

    Ok(())
}

/// Returns an S3 client with retry configuration.
pub async fn get_aws_clients(config: &Config) -> Result<AwsClients> {
    let aws_clients = AwsClients::new(config)
        .await
        .expect("failed to create aws clients");
    Ok(aws_clients)
}
