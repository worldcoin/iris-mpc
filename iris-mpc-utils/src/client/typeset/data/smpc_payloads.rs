use serde::{Deserialize, Serialize};

use iris_mpc_common::helpers::{smpc_request, smpc_response};

/// Enumeration over request messages enqueued upon system ingress queue.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestPayload {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}

/// Enumeration over response messages dequeued from system egress queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsePayload {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::ResetCheckResult),
    ResetUpdate(smpc_response::ResetUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}

impl ResponsePayload {
    pub fn node_id(&self) -> usize {
        match self {
            Self::IdentityDeletion(result) => result.node_id,
            Self::Reauthorization(result) => result.node_id,
            Self::ResetCheck(result) => result.node_id,
            Self::ResetUpdate(result) => result.node_id,
            Self::Uniqueness(result) => result.node_id,
        }
    }

    /// Returns true if the response indicates a processing error.
    pub fn is_error(&self) -> bool {
        match self {
            Self::IdentityDeletion(result) => !result.success,
            Self::Reauthorization(result) => result.error.unwrap_or(false),
            Self::ResetCheck(result) => result.error.unwrap_or(false),
            Self::ResetUpdate(_) => false,
            Self::Uniqueness(result) => result.error.unwrap_or(false),
        }
    }

    /// Returns the error reason string if one is available.
    pub fn error_reason(&self) -> Option<&str> {
        match self {
            Self::IdentityDeletion(_) => None,
            Self::Reauthorization(result) => result.error_reason.as_deref(),
            Self::ResetCheck(result) => result.error_reason.as_deref(),
            Self::ResetUpdate(_) => None,
            Self::Uniqueness(result) => result.error_reason.as_deref(),
        }
    }
}
