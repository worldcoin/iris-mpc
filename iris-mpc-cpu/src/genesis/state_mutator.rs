use super::utils::{
    self,
    aws::{get_s3_bucket_for_iris_deletions, get_s3_key_for_iris_deletions, IrisDeletionsForS3},
    constants::{
        STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID, STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
    },
    errors::IndexationError,
};
use crate::{hawkers::aby3::aby3_store::Aby3Store, hnsw::graph::graph_store::GraphPg};
use aws_sdk_s3::{primitives::ByteStream as S3_ByteStream, Client as S3_Client};
use eyre::Result;
use iris_mpc_common::{config::Config, IrisSerialId};
use serde::Serialize;
use sqlx::{Postgres, Transaction};
use std::fmt::Debug;

// Component name for logging purposes.
const COMPONENT: &str = "State-Mutator";

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
    config: &Config,
    s3_client: &S3_Client,
    deletions: Vec<IrisSerialId>,
) -> Result<(), IndexationError> {
    // Set bucket/key based on environment.
    let s3_bucket = get_s3_bucket_for_iris_deletions(config);
    let s3_key = get_s3_key_for_iris_deletions(config);
    utils::log_info(
        COMPONENT,
        format!(
            "Inserting deleted serial ids into S3 bucket: {}, key: {}",
            s3_bucket, s3_key
        ),
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
            utils::log_error(
                COMPONENT,
                format!("Failed to download file from S3: {}", err),
            );
            IndexationError::AwsS3ObjectUpload
        })?;

    Ok(())
}

/// Sets serial id of last Iris to have been indexed.
///
/// # Arguments
///
/// * `tx` - PostgreSQL transaction to use for operation scope.
/// * `value` - Iris serial id to be persisted.
///
/// # Returns
///
/// Result<()> on success
///
pub async fn set_last_indexed_iris_id(
    tx: &mut Transaction<'_, Postgres>,
    value: IrisSerialId,
) -> Result<(), IndexationError> {
    set_state_element(tx, STATE_KEY_LAST_INDEXED_IRIS_ID, &value).await
}

/// Sets the last indexed modification id.
///
/// # Arguments
///
/// * `tx` - PostgreSQL transaction to use for operation scope.
/// * `value` - Modification id to be persisted.
///
/// # Returns
///
/// Result<()> on success
///
pub async fn set_last_indexed_modification_id(
    tx: &mut Transaction<'_, Postgres>,
    value: i64,
) -> Result<(), IndexationError> {
    set_state_element(tx, STATE_KEY_LAST_INDEXED_MODIFICATION_ID, &value).await
}

/// Persists a state element to remote store.
async fn set_state_element<T: Serialize + Debug>(
    tx: &mut Transaction<'_, Postgres>,
    key: &str,
    value: &T,
) -> Result<(), IndexationError> {
    utils::log_info(
        COMPONENT,
        format!(
            "Persisting Genesis indexation state element: key={} :: value={:?}",
            key, value
        ),
    );

    GraphPg::<Aby3Store>::set_persistent_state(tx, STATE_DOMAIN, key, &value)
        .await
        .map_err(|err| {
            IndexationError::PersistIndexationStateElement(key.to_string(), err.to_string())
        })?;

    Ok(())
}
