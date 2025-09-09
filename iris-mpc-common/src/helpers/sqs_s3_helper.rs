use crate::helpers::key_pair::SharesDecodingError;
use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client,
};

pub async fn upload_file_to_s3(
    bucket: &str,
    key: &str,
    s3_client: Client,
    contents: &[u8],
) -> Result<String, SharesDecodingError> {
    let content_bytestream = ByteStream::new(SdkBody::from(contents));

    // Create a PutObject request
    match s3_client
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
    Ok(key.to_string())
}
