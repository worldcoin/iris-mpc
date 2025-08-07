use serde::{Deserialize, Serialize};

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
#[derive(Clone, Serialize, Deserialize)]
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
            true => "COMPLETED",
            _ => "IN_PROGRESS",
        }
    }
}

impl ModificationType {
    pub fn to_str(&self) -> &'static str {
        match self {
            ModificationType::ResetUpdate => "reset_update",
            ModificationType::Reauth => "reauth",
            ModificationType::Uniqueness => "uniqueness",
        }
    }
}
