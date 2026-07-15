use super::utils::errors::IndexationError;
use crate::{
    execution::hawk_main::HawkOps, hawkers::aby3::aby3_store::Aby3Store,
    hnsw::graph::graph_store::GraphPg,
};
use eyre::Result;
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};
use iris_mpc_common::{config::Config, helpers::sync::Modification, SerialId};
use iris_mpc_store::Store;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sqlx::{Postgres, Transaction};
use std::{fmt::Debug, sync::Arc};

// The below constants are public for the test_utils module

/// Domain for persistent state store entry for last indexed id
pub const STATE_DOMAIN: &str = "genesis";

/// Key for persistent state store entry for last indexed iris id
pub const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";

/// Key for persistent state store entry for last indexed modification id
pub const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";

/// Fetches serial identifiers marked as deleted.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `s3_client` - A configured object-store client.
/// * `max_indexation_id` - Maximum Iris serial identifier to be indexed.
///
/// # Returns
///
/// A set of Iris serial identifiers marked as deleted.
///
pub async fn get_iris_deletions(
    config: &Config,
    s3_client: &ObjectStoreClient,
    max_indexation_id: SerialId,
) -> Result<Vec<SerialId>, IndexationError> {
    // Struct for deserialization.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    struct S3Object {
        deleted_serial_ids: Vec<SerialId>,
    }

    let store_location = config.iris_deletions_store_location();
    let object_key = format!("{}_deleted_serial_ids.json", config.environment);
    tracing::info!(
        "Fetching deleted serial ids from object store: {}, key: {}",
        store_location,
        object_key
    );

    // Fetch from object storage.
    let store = s3_client.store(&store_location).map_err(|err| {
        tracing::error!("Failed to construct object store: {err}");
        IndexationError::AwsS3ObjectDownload
    })?;
    let location = path(&object_key).map_err(|err| {
        tracing::error!("Invalid object key: {err}");
        IndexationError::AwsS3ObjectDownload
    })?;
    let response = store.get(&location).await.map_err(|err| {
        tracing::error!("Failed to download file from object store: {}", err);
        IndexationError::AwsS3ObjectDownload
    })?;

    let s3_object_bytes = response.bytes().await.map_err(|err| {
        tracing::error!("Failed to get object body: {}", err);
        IndexationError::AwsS3ObjectDeserialize
    })?;

    let S3Object { deleted_serial_ids } =
        serde_json::from_slice(&s3_object_bytes).map_err(|err| {
            tracing::error!("Failed to deserialize S3 object: {}", err);
            IndexationError::AwsS3ObjectDeserialize
        })?;

    // Return those <= max indexation id.
    Ok(deleted_serial_ids
        .iter()
        .filter(|&&x| x <= max_indexation_id)
        .cloned()
        .collect::<Vec<u32>>())
}

/// Retrieves Iris modifications that need to be applied post indexation phase one.
/// N.B. The modifications are returned in ascending order of the serial id.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `after_modification_id` - Set of Iris serial identifiers within batch.
/// * `serial_id_less_than` - Set of Iris serial identifiers within batch.
///
/// # Returns
///
/// 2 member tuple: (Modification data, Max completed modification from the table).
///
pub async fn get_iris_modifications(
    iris_store: &Store,
    from_modification_id: i64,
    to_serial_id: u32,
) -> Result<(Vec<Modification>, i64), IndexationError> {
    tracing::info!(
        "Fetching Iris modifications for indexation: from-modification-id={} :: to-serial-id={}",
        from_modification_id,
        to_serial_id
    );

    let (modifications, max_id) = iris_store
        .get_persisted_modifications_after_id(from_modification_id, to_serial_id)
        .await
        .map_err(|err| IndexationError::FetchModificationBatch(err.to_string()))?;
    let max_id = max_id.unwrap_or(0);
    Ok((modifications, max_id))
}

/// Get serial id of last iris to have been indexed.
///
/// # Arguments
///
/// * `graph_store` - Graph PostgreSQL store provider.
///
/// # Returns
///
/// The serial id of last iris to have been indexed, or 0 if none has been recorded.
///
pub async fn get_last_indexed_iris_id(
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
) -> Result<SerialId> {
    get_state_element(graph_store, STATE_KEY_LAST_INDEXED_IRIS_ID).await
}

/// Gets the modification id of the last indexed modification.
///
/// # Arguments
///
/// * `graph_store` -Arc of Graph PostgreSQL store provider.
///
/// # Returns
///
/// The modification id of the last indexed modification, or 0 if none has been recorded.
///
pub async fn get_last_indexed_modification_id(
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
) -> Result<i64> {
    get_state_element(graph_store, STATE_KEY_LAST_INDEXED_MODIFICATION_ID).await
}

/// Gets a state element value from remote store.
async fn get_state_element<T: DeserializeOwned + Default>(
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
    key: &str,
) -> Result<T> {
    tracing::info!("Retrieving Genesis indexation state element: key={}", key);

    let value = graph_store
        .get_persistent_state(STATE_DOMAIN, key)
        .await?
        .unwrap_or(T::default());

    Ok(value)
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
    value: SerialId,
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
    tracing::info!(
        "Persisting Genesis indexation state element: key={} :: value={:?}",
        key,
        value
    );

    GraphPg::<Aby3Store>::set_persistent_state(tx, STATE_DOMAIN, key, &value)
        .await
        .map_err(|err| {
            IndexationError::PersistIndexationStateElement(key.to_string(), err.to_string())
        })?;

    Ok(())
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
pub async fn unset_last_indexed_iris_id(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<(), IndexationError> {
    unset_state_element(tx, STATE_KEY_LAST_INDEXED_IRIS_ID).await
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
pub async fn unset_last_indexed_modification_id(
    tx: &mut Transaction<'_, Postgres>,
) -> Result<(), IndexationError> {
    unset_state_element(tx, STATE_KEY_LAST_INDEXED_MODIFICATION_ID).await
}

/// Unsets an indexation state element.
///
/// # Arguments
///
/// * `tx` - PostgreSQL transaction to use for operation scope.
/// * `key` - Key of state element to be unset.
///
/// # Returns
///
/// Result<()> on success
///
async fn unset_state_element(
    tx: &mut Transaction<'_, Postgres>,
    key: &str,
) -> Result<(), IndexationError> {
    tracing::info!("Unsetting Genesis indexation state element: key={}", key);

    GraphPg::<Aby3Store>::delete_persistent_state(tx, STATE_DOMAIN, key)
        .await
        .map_err(|err| {
            IndexationError::UnsetIndexationStateElement(key.to_string(), err.to_string())
        })?;

    Ok(())
}

#[cfg(test)]
mod object_store_tests {
    use super::get_iris_deletions;
    use iris_mpc_common::{
        config::Config,
        object_store::{path, ObjectStoreClient, ObjectStoreExt},
    };

    #[tokio::test]
    async fn reads_deletions_from_configured_object_store_location() {
        let store_location = "memory:///deletion-snapshots";
        let config: Config = serde_json::from_value(serde_json::json!({
            "environment": "test",
            "iris_deletions_store_location": store_location
        }))
        .unwrap();
        let client = ObjectStoreClient::new(None, false);
        let store = client.store(store_location).unwrap();
        store
            .put(
                &path("test_deleted_serial_ids.json").unwrap(),
                serde_json::to_vec(&serde_json::json!({
                    "deleted_serial_ids": [1, 3, 8]
                }))
                .unwrap()
                .into(),
            )
            .await
            .unwrap();

        let deletions = get_iris_deletions(&config, &client, 5).await.unwrap();

        assert_eq!(deletions, vec![1, 3]);
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use crate::genesis::state_accessor::unset_last_indexed_modification_id;

    use super::{
        get_last_indexed_iris_id, get_last_indexed_modification_id, set_last_indexed_iris_id,
        set_last_indexed_modification_id, unset_last_indexed_iris_id,
    };
    use crate::{
        execution::hawk_main::HawkOps, hawkers::aby3::aby3_store::Aby3Store,
        hnsw::graph::graph_store::GraphPg,
    };
    use eyre::Result;
    use iris_mpc_common::postgres::{AccessMode, PostgresClient, PostgresSchemaName};
    use iris_mpc_store::{
        test_utils::{cleanup, temporary_name, test_db_url},
        Store as IrisStore,
    };
    use std::sync::Arc;

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;

    // Returns a set of test resources.
    async fn get_resources() -> Result<(
        IrisStore,
        Arc<GraphPg<Aby3Store<HawkOps>>>,
        PostgresClient,
        PostgresSchemaName,
    )> {
        // Set PostgreSQL client + store.
        let pg_schema = temporary_name();
        let pg_client =
            PostgresClient::new(&test_db_url()?, &pg_schema, AccessMode::ReadWrite).await?;
        // Set graph store
        let graph_store = GraphPg::new(&pg_client).await?;
        let graph_store_arc = Arc::new(graph_store);

        // Set iris store.
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

        Ok((iris_store, graph_store_arc, pg_client, pg_schema))
    }

    #[tokio::test]
    async fn test_last_indexed_iris_id() -> Result<()> {
        // Set resources.
        let (_iris_store, graph_store, pg_client, pg_schema) = get_resources().await.unwrap();

        // Get -> should be zero.
        let id_of_last_indexed = get_last_indexed_iris_id(graph_store.clone()).await?;
        assert_eq!(id_of_last_indexed, 0);

        // Set -> 10.
        let id_of_last_indexed = 10_u32;
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        set_last_indexed_iris_id(&mut tx, id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 10.
        let id_of_last_indexed = get_last_indexed_iris_id(graph_store.clone()).await?;
        assert_eq!(id_of_last_indexed, 10);

        // Unset.
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        unset_last_indexed_iris_id(&mut tx).await?;
        tx.commit().await?;

        // Get -> should be 0.
        let id_of_last_indexed = get_last_indexed_iris_id(graph_store.clone()).await?;
        assert_eq!(id_of_last_indexed, 0);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_last_indexed_modificiation_id() -> Result<()> {
        // Set resources.
        let (_iris_store, graph_store, pg_client, pg_schema) = get_resources().await.unwrap();

        // Get -> should be zero.
        let modification_id_of_last_indexed =
            get_last_indexed_modification_id(graph_store.clone()).await?;
        assert_eq!(modification_id_of_last_indexed, 0);

        // Set -> 42.
        let modification_id_of_last_indexed = 42_i64;
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        set_last_indexed_modification_id(&mut tx, modification_id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 42.
        let modification_id_of_last_indexed =
            get_last_indexed_modification_id(graph_store.clone()).await?;
        assert_eq!(modification_id_of_last_indexed, 42);

        // Set -> 999.
        let modification_id_of_last_indexed = 999_i64;
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        set_last_indexed_modification_id(&mut tx, modification_id_of_last_indexed).await?;
        tx.commit().await?;

        // Get -> should be 999.
        let modification_id_of_last_indexed =
            get_last_indexed_modification_id(graph_store.clone()).await?;
        assert_eq!(modification_id_of_last_indexed, 999);

        // Unset.
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        unset_last_indexed_modification_id(&mut tx).await?;
        tx.commit().await?;

        // Get -> should be 0.
        let modification_id_of_last_indexed =
            get_last_indexed_modification_id(graph_store.clone()).await?;
        assert_eq!(modification_id_of_last_indexed, 0);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }
}
