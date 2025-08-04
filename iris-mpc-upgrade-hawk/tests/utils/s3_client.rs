use crate::utils::errors::TestError;
use aws_sdk_s3::{primitives::ByteStream as S3_ByteStream, Client as S3_Client};
use eyre::Result;
use iris_mpc::services::aws::clients::AwsClients;
use iris_mpc_common::{config::Config, IrisSerialId};
use iris_mpc_cpu::genesis::state_accessor::S3Object;

/// Uploads to an AWS S3 bucket a set of serial identifiers marked as deleted.
pub async fn upload_iris_deletions(
    data: &Vec<IrisSerialId>,
    s3_client: &S3_Client,
    config: &Config,
) -> Result<(), TestError> {
    // Set bucket/key based on environment.
    let s3_bucket = get_s3_bucket_for_iris_deletions(&config.environment);
    let s3_key = get_s3_key_for_iris_deletions(&config.environment);

    let obj = S3Object {
        deleted_serial_ids: data.to_owned(),
    };
    let obj_bytes = serde_json::to_vec(&obj).unwrap();
    let body = S3_ByteStream::from(obj_bytes);

    s3_client
        .put_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .body(body)
        .send()
        .await
        .unwrap();

    Ok(())
}

/// Returns an S3 client with retry configuration.
pub async fn get_aws_clients(config: &Config) -> Result<AwsClients> {
    let aws_clients = AwsClients::new(&config)
        .await
        .expect("failed to create aws clients");
    Ok(aws_clients)
}

/// AWS S3 bucket for iris deletions.
pub fn get_s3_bucket_for_iris_deletions(environment: &str) -> String {
    format!("wf-smpcv2-{}-sync-protocol", environment)
}

/// AWS S3 key for iris deletions.
pub fn get_s3_key_for_iris_deletions(environment: &str) -> String {
    format!("{}_deleted_serial_ids.json", environment)
}
