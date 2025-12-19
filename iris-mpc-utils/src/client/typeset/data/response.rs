use iris_mpc_common::helpers::smpc_response;

/// Enumeration over system responses dequeued from system egress queue.
#[derive(Clone, Debug)]
pub enum ResponseBody {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::ResetCheckResult),
    ResetUpdate(smpc_response::ResetUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}

impl ResponseBody {
    pub fn node_id(&self) -> usize {
        match self {
            Self::IdentityDeletion(result) => result.node_id,
            Self::Reauthorization(result) => result.node_id,
            Self::ResetCheck(result) => result.node_id,
            Self::ResetUpdate(result) => result.node_id,
            Self::Uniqueness(result) => result.node_id,
        }
    }

    pub fn iris_serial_id(&self) -> Option<u32> {
        match self {
            Self::IdentityDeletion(result) => Some(result.serial_id),
            Self::Reauthorization(result) => Some(result.serial_id),
            Self::ResetCheck(_) => None,
            Self::ResetUpdate(result) => Some(result.serial_id),
            Self::Uniqueness(result) => result.serial_id,
        }
    }
}
