use super::super::errors::IndexationError;
use aws_sdk_s3::{config::Region as S3_Region, Client as S3_CLient};
use iris_mpc_common::config::Config;
use rand::prelude::*;

// Name of S3 bucket into which genesis indexation information has been placed.
// const AWS_S3_BUCKET_NAME: &str = "wf-smpcv1-hnsw-genesis-indexation-info";

/// Fetches V1 serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - System configuration information.
///
/// # Returns
///
/// A set of V1 serial identifiers marked as deleted.
///
pub(crate) async fn fetch_iris_v1_deletions(config: &Config) -> Result<Vec<i64>, IndexationError> {
    // Destructure AWS configuration settings.
    let aws_endpoint = config
        .aws
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?
        .endpoint
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;
    let aws_region = config
        .aws
        .as_ref()
        .unwrap()
        .region
        .as_ref()
        .ok_or(IndexationError::AwsConfigurationError)?;

    // Set AWS S3 client.
    let aws_config = aws_config::from_env()
        .region(S3_Region::new(aws_region.clone()))
        .load()
        .await;
    let s3_cfg = aws_sdk_s3::config::Builder::from(&aws_config)
        .endpoint_url(aws_endpoint.clone())
        .force_path_style(true)
        .build();
    let _ = S3_CLient::from_conf(s3_cfg);

    // Set AWS S# response.
    // TODO: test once resource has been deployed
    // let s3_response = s3_client
    //     .get_object()
    //     .bucket(AWS_S3_BUCKET_NAME)
    //     .key("TODO: get-key")
    //     .send()
    //     .await
    //     .map_err(|err| {
    //         tracing::error!("Failed to download file from AWS S3: {:?}", err);
    //         IndexationError::AwsS3DownloadError
    //     })?;
    // let _ = s3_response.body.collect().await.map_err(|err| {
    //     tracing::error!("Failed to decode file from AWS S3: {}", err);
    //     IndexationError::AwsS3DownloadError
    // })?;

    // TODO: Implement fetching V1 serial identifiers marked as deleted from AWS S3.
    let mut rng = rand::thread_rng();
    let mut identifiers = (1_i64..1000_i64).choose_multiple(&mut rng, 50);
    identifiers.sort();

    Ok(identifiers)
}
