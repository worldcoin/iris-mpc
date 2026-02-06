use serde::{Deserialize, Serialize};

use iris_mpc_common::helpers::{smpc_request, smpc_response};

use super::super::errors::ServiceClientError;

/// Enumeration over request messages enqueued upon system ingress queue.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum RequestPayload {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}

/// Enumeration over response messages dequeued from system egress queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum ResponsePayload {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::ResetCheckResult),
    ResetUpdate(smpc_response::ResetUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}

impl ResponsePayload {
    pub(super) fn node_id(&self) -> usize {
        match self {
            Self::IdentityDeletion(result) => result.node_id,
            Self::Reauthorization(result) => result.node_id,
            Self::ResetCheck(result) => result.node_id,
            Self::ResetUpdate(result) => result.node_id,
            Self::Uniqueness(result) => result.node_id,
        }
    }

    /// Validates the response, returning an error if the response indicates failure.
    pub(crate) fn validate(&self) -> Result<(), ServiceClientError> {
        let (has_error, error_reason) = match self {
            Self::IdentityDeletion(result) => (!result.success, None),
            Self::Reauthorization(result) => {
                (result.error.unwrap_or(false), result.error_reason.as_deref())
            }
            Self::ResetCheck(result) => {
                (result.error.unwrap_or(false), result.error_reason.as_deref())
            }
            Self::ResetUpdate(_) => (false, None),
            Self::Uniqueness(result) => {
                (result.error.unwrap_or(false), result.error_reason.as_deref())
            }
        };

        if has_error {
            let reason = error_reason.unwrap_or("unknown error");
            Err(ServiceClientError::ResponseError(format!(
                "{}: {:?}",
                reason, self
            )))
        } else {
            Ok(())
        }
    }
}
