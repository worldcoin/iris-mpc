use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::{presigning::PresigningConfig, primitives::ByteStream, Client, Error};
use std::time::Duration;

pub async fn upload_file_and_generate_presigned_url(
    bucket: &str,
    key: &str,
    contents: &[u8],
) -> Result<String, Error> {
    // Load AWS configuration
    let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");
    let config = aws_config::from_env().region(region_provider).load().await;

    // Create S3 client
    let client = Client::new(&config);

    // Create a PutObject request
    client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(ByteStream::from_static(contents))
        .send()
        .await
        .expect("Failed to upload file.");

    println!("File uploaded successfully.");

    // Create a presigned URL for the uploaded file
    let presigning_config = PresigningConfig::expires_in(Duration::from_secs(3600))?;
    let presigned_req = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .presigned(presigning_config)
        .await?;

    // Return the presigned URL
    Ok(presigned_req.uri().to_string())
}
