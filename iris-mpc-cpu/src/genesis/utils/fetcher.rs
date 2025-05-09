use super::{errors::IndexationError, types::IrisSerialId};
use aws_sdk_s3::Client as S3_Client;
use iris_mpc_store::{DbStoredIris, Store as IrisPgresStore};
use serde::{Deserialize, Serialize};

/// Fetches height of indexed from store.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Height of indexed Iris's.
///
pub(crate) async fn fetch_height_of_indexed(
    _: &IrisPgresStore,
) -> Result<IrisSerialId, IndexationError> {
    // TODO: fetch from store.
    Ok(1)
}

/// Fetches height of protocol from store.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Height of stored Iris's.
///
pub(crate) async fn fetch_height_of_protocol(
    iris_store: &IrisPgresStore,
) -> Result<IrisSerialId, IndexationError> {
    iris_store
        .count_irises()
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisById)
        .map(|val| val as IrisSerialId)
}

/// Fetch a batch of iris data for indexation.
///
/// # Arguments
///
/// * `store` - Iris PostgreSQL store provider.
/// * `identifiers` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// Iris data for indexation.
///
pub(crate) async fn fetch_iris_batch(
    iris_store: &IrisPgresStore,
    identifiers: Vec<IrisSerialId>,
) -> Result<Vec<DbStoredIris>, IndexationError> {
    let data = iris_store
        .fetch_iris_batch(identifiers)
        .await
        .map_err(|_| IndexationError::PostgresFetchIrisBatch)
        .unwrap();

    Ok(data)
}

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `s3_client` - A configured AWS S3 client instance.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
#[allow(dead_code)]
pub(crate) async fn fetch_iris_deletions(
    s3_client: &S3_Client,
    env: String,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    // Struct for deserialization.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct S3Object {
        deleted_serial_ids: Vec<IrisSerialId>,
    }

    // Compose bucket and key based on environment
    let bucket = format!("wf-smpcv2-{}-sync-protocol", env);
    let key = format!("{}_deleted_serial_ids.json", env);

    tracing::info!(
        "Fetching deleted serial ids from S3 bucket: {}, key: {}",
        bucket,
        key
    );

    // Fetch from S3.
    let s3_response = s3_client
        .get_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
        .map_err(|err| {
            tracing::error!("Failed to download file from S3: {}", err);
            IndexationError::AwsS3ObjectDownload
        })?;

    // Consume S3 object stream.
    let s3_object_body = s3_response.body.collect().await.map_err(|e| {
        tracing::error!("Failed to get object body: {}", e);
        IndexationError::AwsS3ObjectDeserialize
    })?;

    // Decode S3 object bytes.
    let s3_object_bytes = s3_object_body.into_bytes();
    let s3_object: S3Object = serde_json::from_slice(&s3_object_bytes).map_err(|err| {
        tracing::error!("Failed to deserialize S3 object: {}", err);
        IndexationError::AwsS3ObjectDeserialize
    })?;

    Ok(s3_object.deleted_serial_ids)
}
