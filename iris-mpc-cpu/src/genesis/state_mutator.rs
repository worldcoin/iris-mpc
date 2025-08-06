use super::utils::{
    self,
    constants::{
        STATE_DOMAIN, STATE_KEY_LAST_INDEXED_IRIS_ID, STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
    },
    errors::IndexationError,
};
use crate::{hawkers::aby3::aby3_store::Aby3Store, hnsw::graph::graph_store::GraphPg};
use eyre::Result;
use iris_mpc_common::IrisSerialId;
use serde::Serialize;
use sqlx::{Postgres, Transaction};
use std::fmt::Debug;

// Component name for logging purposes.
const COMPONENT: &str = "State-Mutator";

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
