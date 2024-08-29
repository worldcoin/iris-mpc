use crate::helpers::key_pair::SharesDecodingError;
use aws_config::{meta::region::RegionProviderChain, retry::RetryConfig};
use aws_sdk_s3::{
    presigning::PresigningConfig,
    primitives::{ByteStream, SdkBody},
    Client,
};
use std::time::Duration;

pub async fn upload_file_and_generate_presigned_url(
    bucket: &str,
    key: &str,
    region: &'static str,
    contents: &[u8],
) -> Result<String, SharesDecodingError> {
    // Load AWS configuration
    let region_provider = RegionProviderChain::first_try(region).or_default_provider();
    let config = aws_config::from_env()
        .region(region_provider)
        .retry_config(RetryConfig::standard().with_max_attempts(5))
        .load()
        .await;

    // Create S3 client
    let client = Client::new(&config);
    let content_bytestream = ByteStream::new(SdkBody::from(contents));

    // Create a PutObject request
    match client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content_bytestream)
        .send()
        .await
    {
        Ok(_) => {
            tracing::info!("File uploaded successfully.");
        }
        Err(e) => {
            tracing::error!("Error: Failed to upload file: {:?}", e);
            return Err(SharesDecodingError::UploadS3Error);
        }
    }

    tracing::info!("File uploaded successfully.");

    // Create a presigned URL for the uploaded file
    let presigning_config = match PresigningConfig::expires_in(Duration::from_secs(36000)) {
        Ok(config) => config,
        Err(e) => return Err(SharesDecodingError::PresigningConfigError(e)),
    };

    let presigned_req = match client
        .get_object()
        .bucket(bucket)
        .key(key)
        .presigned(presigning_config)
        .await
    {
        Ok(req) => req,
        Err(e) => return Err(SharesDecodingError::PresignedRequestError(e)),
    };

    // Return the presigned URL
    Ok(presigned_req.uri().to_string())
}
