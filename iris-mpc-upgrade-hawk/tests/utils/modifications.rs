use std::{fmt::Display, ops::DerefMut};

use eyre::Result;
use iris_mpc_common::helpers::{
    smpc_request::{REAUTH_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    sync::{MOD_STATUS_COMPLETED, MOD_STATUS_IN_PROGRESS},
};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, Transaction};

// from iris-mpc-common/helpers/smpc_request.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModificationType {
    /// searches for a new iris scan and if it finds a match, overwrites the iris code
    /// in the system with the new scan. AKA reset-check plus overwrite
    ResetUpdate,
    /// lets the user update their iris scan to a new version
    Reauth,
    /// ignored by genesis
    Uniqueness,
}

/// used as inputs to iris-mpc-store > insert_modification()
/// note that s3_url, result_message_body, and graph_mutation (from the Modification struct) can all be None
/// look for JobRequest::Modification in iris-mpc-cpu/src/genesis/hawk_handle.rs to see how these are handled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationInput {
    /// needs to match an existing vector id
    pub serial_id: i64,
    pub request_type: ModificationType,
    pub completed: bool,
    pub persisted: bool,
}

impl ModificationInput {
    pub fn new(
        serial_id: i64,
        request_type: ModificationType,
        completed: bool,
        persisted: bool,
    ) -> Self {
        Self {
            serial_id,
            request_type,
            completed,
            persisted,
        }
    }

    pub fn from_slice(inputs: &[(i64, ModificationType, bool, bool)]) -> Vec<Self> {
        inputs
            .iter()
            .map(|(serial_id, request_type, completed, persisted)| {
                Self::new(*serial_id, request_type.clone(), *completed, *persisted)
            })
            .collect()
    }

    pub fn get_status(&self) -> &'static str {
        match self.completed {
            true => MOD_STATUS_COMPLETED,
            false => MOD_STATUS_IN_PROGRESS,
        }
    }
}

impl ModificationType {
    pub fn is_updating(&self) -> bool {
        matches!(
            self,
            ModificationType::ResetUpdate | ModificationType::Reauth
        )
    }
}

impl Display for ModificationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match self {
            ModificationType::ResetUpdate => RESET_UPDATE_MESSAGE_TYPE,
            ModificationType::Reauth => REAUTH_MESSAGE_TYPE,
            ModificationType::Uniqueness => UNIQUENESS_MESSAGE_TYPE,
        };
        write!(f, "{}", variant)
    }
}

/// Test functionality which updates an iris only by incrementing its version,
/// without changing the underlying iris code.
pub async fn increment_iris_version(
    tx: &mut Transaction<'_, Postgres>,
    serial_id: i64,
) -> Result<()> {
    let query = sqlx::query(
        r#"
        UPDATE irises SET version_id = version_id + 1
        WHERE id = $1;
        "#,
    )
    .bind(serial_id);
    query.execute(tx.deref_mut()).await?;

    Ok(())
}

pub async fn persist_modification(
    tx: &mut Transaction<'_, Postgres>,
    modification_id: i64,
) -> Result<()> {
    let query = sqlx::query(
        r#"
        UPDATE modifications SET status = 'COMPLETED', persisted = true
        WHERE id = $1;
        "#,
    )
    .bind(modification_id);
    query.execute(tx.deref_mut()).await?;

    Ok(())
}
