use crate::helpers::key_pair::SharesDecodingError;
use crate::object_store::{path, ObjectStoreClient, ObjectStoreExt};

pub async fn upload_file_to_s3(
    bucket: &str,
    key: &str,
    object_store_client: ObjectStoreClient,
    contents: &[u8],
) -> Result<String, SharesDecodingError> {
    let store = object_store_client.store(bucket).map_err(|e| {
        tracing::error!("Failed to create object store for {bucket}: {e}");
        SharesDecodingError::ObjectStoreUploadError
    })?;
    let location = path(key).map_err(|e| {
        tracing::error!("Invalid object key {key}: {e}");
        SharesDecodingError::ObjectStoreUploadError
    })?;

    match store.put(&location, contents.to_vec().into()).await {
        Ok(_) => {
            tracing::info!("File {key} uploaded to object store successfully");
        }
        Err(e) => {
            tracing::error!("Failed to upload file {key}: {e:?}");
            return Err(SharesDecodingError::ObjectStoreUploadError);
        }
    }
    Ok(key.to_string())
}
