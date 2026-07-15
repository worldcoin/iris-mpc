use eyre::{eyre, Result};
use iris_mpc::services::aws::clients::AwsClients;
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};
use iris_mpc_common::{config::Config, SerialId};
use serde::Serialize;

/// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
pub async fn upload_iris_deletions(
    data: &Vec<SerialId>,
    s3: &ObjectStoreClient,
    environment: &str,
) -> Result<()> {
    // Set bucket/key based on environment.
    let s3_bucket = get_s3_bucket_for_iris_deletions(environment);
    let s3_key = get_s3_key_for_iris_deletions(environment);
    tracing::info!(
        "Inserting deleted serial ids into S3 bucket: {}, key: {}",
        s3_bucket,
        s3_key
    );

    // Set body of payload to be persisted.
    let body = serde_json::to_string(&IrisDeletionsForS3 {
        deleted_serial_ids: data.to_owned(),
    })
    .unwrap()
    .into_bytes();

    // Upload payload.
    s3.store(&s3_bucket)?
        .put(&path(&s3_key)?, body.into())
        .await
        .map_err(|err| {
            tracing::error!("Failed to upload file to S3: {}", err);
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

// Struct for S3 serialization.
#[derive(Serialize, Debug, Clone)]
pub struct IrisDeletionsForS3 {
    pub deleted_serial_ids: Vec<SerialId>,
}

/// AWS S3 bucket for iris deletions.
pub fn get_s3_bucket_for_iris_deletions(environment: &str) -> String {
    format!("wf-smpcv2-{}-sync-protocol", environment)
}

/// AWS S3 key for iris deletions.
pub fn get_s3_key_for_iris_deletions(environment: &str) -> String {
    format!("{}_deleted_serial_ids.json", environment)
}
