use std::fmt::Display;

use iris_mpc_common::helpers::{
    smpc_request::{
        REAUTH_MESSAGE_TYPE, RECOVERY_UPDATE_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
        UNIQUENESS_MESSAGE_TYPE,
    },
    sync::{Modification, MOD_STATUS_COMPLETED, MOD_STATUS_IN_PROGRESS},
};
use serde::{Deserialize, Serialize};

// from iris-mpc-common/helpers/smpc_request.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModificationType {
    /// searches for a new iris scan and if it finds a match, overwrites the iris code
    /// in the system with the new scan. AKA reset-check plus overwrite
    ResetUpdate,
    /// similar to `ResetUpdate` but triggered via recovery flow
    RecoveryUpdate,
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
    pub mod_id: i64,
    /// needs to match an existing vector id
    pub serial_id: i64,
    pub request_type: ModificationType,
    pub completed: bool,
    pub persisted: bool,
}

impl ModificationInput {
    pub const fn new(
        mod_id: i64,
        serial_id: i64,
        request_type: ModificationType,
        completed: bool,
        persisted: bool,
    ) -> Self {
        Self {
            mod_id,
            serial_id,
            request_type,
            completed,
            persisted,
        }
    }

    pub fn get_status(&self) -> &'static str {
        match self.completed {
            true => MOD_STATUS_COMPLETED,
            false => MOD_STATUS_IN_PROGRESS,
        }
    }

    pub fn is_finalized(&self) -> bool {
        self.completed && self.persisted
    }
}

impl From<ModificationInput> for Modification {
    fn from(value: ModificationInput) -> Self {
        Modification {
            id: value.mod_id,
            serial_id: Some(value.serial_id),
            request_type: value.request_type.to_string(),
            s3_url: None,
            status: value.get_status().to_string(),
            persisted: value.persisted,
            result_message_body: None,
            graph_mutation: None,
        }
    }
}

impl ModificationType {
    pub fn is_updating(&self) -> bool {
        matches!(
            self,
            ModificationType::ResetUpdate | ModificationType::RecoveryUpdate | ModificationType::Reauth
        )
    }
}

impl Display for ModificationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = match self {
            ModificationType::ResetUpdate => RESET_UPDATE_MESSAGE_TYPE,
            ModificationType::RecoveryUpdate => RECOVERY_UPDATE_MESSAGE_TYPE,
            ModificationType::Reauth => REAUTH_MESSAGE_TYPE,
            ModificationType::Uniqueness => UNIQUENESS_MESSAGE_TYPE,
        };
        write!(f, "{}", variant)
    }
}

/// Validate that one list of modifications is a valid extension to another list.  For
/// each modification in `start`, checks:
/// - A modification with equal mod_id is still present in `end`
/// - This modification has the same type as before
/// - If the modification starts as persisted and completed, then it remains this way
pub fn modifications_extension_is_valid(
    start: &[ModificationInput],
    end: &[ModificationInput],
) -> bool {
    for m_start in start {
        let m_end = end.iter().find(|m| m.mod_id == m_start.mod_id);
        if let Some(m_end) = m_end {
            if m_end.request_type != m_start.request_type {
                return false;
            }
            if !m_end.is_finalized() && m_start.is_finalized() {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

/// Return a sorted list of serial ids of irises that must be updated (e.g. their version incremented)
/// to reflect changes between the start and end modifications tables.
pub fn modifications_extension_updates(
    start: &[ModificationInput],
    end: &[ModificationInput],
) -> Vec<i64> {
    let mut end: Vec<_> = end.into();

    end.sort_by_key(|m| m.mod_id);
    end.into_iter()
        .filter_map(|m| {
            // m must be finalized and updating
            if m.is_finalized() && m.request_type.is_updating() {
                // but no update if m was already finalized previously
                if let Some(m_start) = start.iter().find(|m_start| m_start.mod_id == m.mod_id) {
                    if m_start.is_finalized() {
                        None
                    } else {
                        Some(m.serial_id)
                    }
                } else {
                    Some(m.serial_id)
                }
            } else {
                None
            }
        })
        .collect()
}
