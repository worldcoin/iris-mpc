use super::utils::{errors::IndexationError, logger};
use aws_sdk_s3::Client as S3_Client;
use eyre::Result;
use iris_mpc_common::{config::Config, helpers::sync::Modification, IrisSerialId};
use iris_mpc_store::{DbStoredIris, Store};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, Transaction};

// Component name for logging purposes.
const COMPONENT: &str = "State-Accessor";

/// Domain for persistent state store entry for last indexed id
const STATE_DOMAIN_GENESIS: &str = "genesis";

/// Key for persistent state store entry for last indexed id
const STATE_KEY_LAST_INDEXED: &str = "id_of_last_indexed";

/// Key for persistent state store entry for last indexed modification id
const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "id_of_last_indexed_modification";

/// Fetch a batch of iris data for indexation.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `identifiers` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// Iris data for indexation.
///
pub async fn fetch_iris_batch(
    iris_store: &Store,
    identifiers: Vec<IrisSerialId>,
) -> Result<Vec<DbStoredIris>, IndexationError> {
    logger::log_info(
        COMPONENT,
        format!(
            "Fetching Iris batch for indexation: batch-size={}",
            identifiers.len()
        ),
    );

    let data = iris_store
        .fetch_iris_batch(identifiers)
        .await
        .map_err(|err| IndexationError::PostgresFetchIrisBatch(err.to_string()))?;

    Ok(data)
}

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `s3_client` - A configured AWS S3 client instance.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
pub async fn fetch_iris_deletions(
    config: &Config,
    s3_client: &S3_Client,
) -> Result<Vec<IrisSerialId>, IndexationError> {
    // Struct for deserialization.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct S3Object {
        deleted_serial_ids: Vec<IrisSerialId>,
    }

    // Set bucket and key based on environment
    let s3_bucket = get_s3_bucket_for_iris_deletions(config.environment.clone());
    let s3_key = get_s3_key_for_iris_deletions(config.environment.clone());
    logger::log_info(
        COMPONENT,
        format!(
            "Fetching deleted serial ids from S3 bucket: {}, key: {}",
            s3_bucket, s3_key
        ),
    );

    // Fetch from S3.
    let s3_response = s3_client
        .get_object()
        .bucket(&s3_bucket)
        .key(&s3_key)
        .send()
        .await
        .map_err(|err| {
            logger::log_error(
                COMPONENT,
                format!("Failed to download file from S3: {}", err),
            );
            IndexationError::AwsS3ObjectDownload
        })?;

    // Consume S3 object stream.
    let s3_object_body = s3_response.body.collect().await.map_err(|err| {
        logger::log_error(COMPONENT, format!("Failed to get object body: {}", err));
        IndexationError::AwsS3ObjectDeserialize
    })?;

    // Decode S3 object bytes.
    let s3_object_bytes = s3_object_body.into_bytes();
    let s3_object: S3Object = serde_json::from_slice(&s3_object_bytes).map_err(|err| {
        logger::log_error(
            COMPONENT,
            format!("Failed to deserialize S3 object: {}", err),
        );
        IndexationError::AwsS3ObjectDeserialize
    })?;

    Ok(s3_object.deleted_serial_ids)
}

/// Fetch Iris modifications that need to be applied post indexation phase one.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `after_modification_id` - Set of Iris serial identifiers within batch.
/// * `serial_id_less_than` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// 2 member tuple: (Modification data, Last modification ID).
///
pub async fn fetch_iris_modifications(
    iris_store: &Store,
    from_modification_id: i64,
    to_serial_id: u32,
) -> Result<(Vec<Modification>, i64), IndexationError> {
    logger::log_info(
        COMPONENT,
        format!(
            "Fetching Iris modifications for indexation: from-modification-id={} :: to-serial-id={}",
            from_modification_id, to_serial_id
        ),
    );

    let data = iris_store
        .get_persisted_modifications_after_id(from_modification_id, to_serial_id)
        .await
        .map_err(|err| IndexationError::PostgresFetchModificationBatch(err.to_string()))?;
    let last_id = data.last().map(|m| m.id).unwrap_or(0);

    Ok((data, last_id))
}

/// Get serial id of last iris to have been indexed.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// Serial id of the last indexed iris, or 0 if no serial id is recorded.
///
pub async fn get_last_indexed_id(iris_store: &Store) -> Result<IrisSerialId> {
    let id = iris_store
        .get_persistent_state(STATE_DOMAIN_GENESIS, STATE_KEY_LAST_INDEXED)
        .await?
        .unwrap_or(0);

    Ok(id)
}

/// Gets the modification id of the last indexed modification.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
///
/// # Returns
///
/// The modification id of the last indexed modification, or 0 if no modification id is recorded.
///
pub async fn get_last_indexed_modification_id(iris_store: &Store) -> Result<i64> {
    let id = iris_store
        .get_persistent_state(STATE_DOMAIN_GENESIS, STATE_KEY_LAST_INDEXED_MODIFICATION_ID)
        .await?
        .unwrap_or(0);

    Ok(id)
}

/// Returns computed name of an S3 bucket for fetching iris deletions.
fn get_s3_bucket_for_iris_deletions(environment: String) -> String {
    format!("wf-smpcv2-{}-sync-protocol", environment)
}

/// Returns computed name of an S3 key for fetching iris deletions.
fn get_s3_key_for_iris_deletions(environment: String) -> String {
    format!("{}_deleted_serial_ids.json", environment)
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
pub async fn set_last_indexed_id(
    tx: &mut Transaction<'_, Postgres>,
    value: IrisSerialId,
) -> Result<()> {
    Store::set_persistent_state(tx, STATE_DOMAIN_GENESIS, STATE_KEY_LAST_INDEXED, &value).await
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
) -> Result<()> {
    Store::set_persistent_state(
        tx,
        STATE_DOMAIN_GENESIS,
        STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
        &value,
    )
    .await
}

/// Unsets serial id of last Iris to have been indexed.
///
/// # Arguments
///
/// * `tx` - PostgreSQL transaction to use for operation scope.
///
/// # Returns
///
/// Result<()> on success
///
pub async fn unset_last_indexed_id(tx: &mut Transaction<'_, Postgres>) -> Result<()> {
    Store::delete_persistent_state(tx, STATE_DOMAIN_GENESIS, STATE_KEY_LAST_INDEXED).await
}

/// Unsets serial id of last Iris to have been indexed.
///
/// # Arguments
///
/// * `tx` - PostgreSQL transaction to use for operation scope.
///
/// # Returns
///
/// Result<()> on success
///
pub async fn unset_last_indexed_modification_id(tx: &mut Transaction<'_, Postgres>) -> Result<()> {
    Store::delete_persistent_state(
        tx,
        STATE_DOMAIN_GENESIS,
        STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
    )
    .await
}

#[cfg(test)]
// #[cfg(feature = "db_dependent")]
mod tests {
    use crate::genesis::state_accessor::unset_last_indexed_modification_id;

    use super::{
        fetch_iris_batch, get_last_indexed_id, get_last_indexed_modification_id,
        set_last_indexed_id, set_last_indexed_modification_id, unset_last_indexed_id,
    };
    use eyre::Result;
    use iris_mpc_common::{
        postgres::{AccessMode, PostgresClient, PostgresSchemaName},
        IrisSerialId,
    };
    use iris_mpc_store::{
        test_utils::{cleanup, temporary_name, test_db_url},
        Store as IrisStore,
    };
    use itertools::Itertools;

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;

    // Returns a set of test resources.
    async fn get_resources() -> Result<(IrisStore, PostgresClient, PostgresSchemaName)> {
        // Set PostgreSQL client + store.
        let pg_schema = temporary_name();
        let pg_client =
            PostgresClient::new(&test_db_url()?, &pg_schema, AccessMode::ReadWrite).await?;

        // Set store.
        let iris_store = IrisStore::new(&pg_client).await?;

        // Set dB with 100 Iris's.
        iris_store
            .init_db_with_random_shares(
                DEFAULT_RNG_SEED,
                DEFAULT_PARTY_ID,
                DEFAULT_SIZE_OF_IRIS_DB,
                true,
            )
            .await?;

        Ok((iris_store, pg_client, pg_schema))
    }

    #[tokio::test]
    async fn test_id_of_last_indexed() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        // Get -> should be zero.
        let id_of_last_indexed = get_last_indexed_id(&iris_store).await?;
        assert_eq!(id_of_last_indexed, 0);

        // Set -> 10.
        let id_of_last_indexed = 10_u32;
        let mut tx = iris_store.tx().await?;
        set_last_indexed_id(&mut tx, id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 10.
        let id_of_last_indexed = get_last_indexed_id(&iris_store).await?;
        assert_eq!(id_of_last_indexed, 10);

        // Unset.
        let mut tx = iris_store.tx().await?;
        unset_last_indexed_id(&mut tx).await?;
        tx.commit().await?;

        // Get -> should be 0.
        let id_of_last_indexed = get_last_indexed_id(&iris_store).await?;
        assert_eq!(id_of_last_indexed, 0);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_fetch_iris_batch() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let identifiers: Vec<IrisSerialId> = (1..11).collect_vec();
        let data = fetch_iris_batch(&iris_store, identifiers).await.unwrap();
        assert_eq!(data.len(), 10);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_modification_id_of_last_indexed() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        // Get -> should be zero.
        let modification_id_of_last_indexed = get_last_indexed_modification_id(&iris_store).await?;
        assert_eq!(modification_id_of_last_indexed, 0);

        // Set -> 42.
        let modification_id_of_last_indexed = 42_i64;
        let mut tx = iris_store.tx().await?;
        set_last_indexed_modification_id(&mut tx, modification_id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 42.
        let modification_id_of_last_indexed = get_last_indexed_modification_id(&iris_store).await?;
        assert_eq!(modification_id_of_last_indexed, 42);

        // Set -> 999.
        let modification_id_of_last_indexed = 999_i64;
        let mut tx = iris_store.tx().await?;
        set_last_indexed_modification_id(&mut tx, modification_id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 999.
        let modification_id_of_last_indexed = get_last_indexed_modification_id(&iris_store).await?;
        assert_eq!(modification_id_of_last_indexed, 999);

        // Unset.
        let mut tx = iris_store.tx().await?;
        unset_last_indexed_modification_id(&mut tx).await?;
        tx.commit().await?;

        // Get -> should be 0.
        let modification_id_of_last_indexed = get_last_indexed_modification_id(&iris_store).await?;
        assert_eq!(modification_id_of_last_indexed, 0);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }
}
