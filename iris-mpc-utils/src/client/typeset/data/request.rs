use std::fmt;

use serde::{Deserialize, Serialize};
use uuid;

use iris_mpc_common::IrisSerialId;

use super::{IrisPairDescriptor, RequestInfo, RequestStatus, ResponsePayload};

/// Encapsulates data pertinent to a system processing request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    IdentityDeletion {
        // Standard request information.
        info: RequestInfo,
        // Serial ID of associated uniqueness request (always known at creation).
        parent: IrisSerialId,
    },
    Reauthorization {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair: Option<IrisPairDescriptor>,
        // Serial ID of associated uniqueness request (always known at creation).
        parent: IrisSerialId,
        // Operation identifier.
        reauth_id: uuid::Uuid,
    },
    ResetCheck {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair: Option<IrisPairDescriptor>,
        // Operation identifier.
        reset_id: uuid::Uuid,
    },
    ResetUpdate {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair: Option<IrisPairDescriptor>,
        // Serial ID of associated uniqueness request (always known at creation).
        parent: IrisSerialId,
        // Operation identifier.
        reset_id: uuid::Uuid,
    },
    Uniqueness {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair: Option<IrisPairDescriptor>,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
    },
}

impl Request {
    pub fn info(&self) -> &RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
        }
    }

    pub fn get_shares_info(&self) -> Option<(uuid::Uuid, Option<IrisPairDescriptor>)> {
        match self {
            Request::IdentityDeletion { .. } => None,
            Request::Reauthorization {
                reauth_id,
                iris_pair,
                ..
            } => Some((*reauth_id, *iris_pair)),
            Request::ResetCheck {
                reset_id,
                iris_pair,
                ..
            } => Some((*reset_id, *iris_pair)),
            Request::ResetUpdate {
                reset_id,
                iris_pair,
                ..
            } => Some((*reset_id, *iris_pair)),
            Request::Uniqueness {
                signup_id,
                iris_pair,
                ..
            } => Some((*signup_id, *iris_pair)),
        }
    }

    fn info_mut(&mut self) -> &mut RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
        }
    }

    #[allow(dead_code)]
    pub fn iris_pair_indexes(&self) -> Vec<usize> {
        match self {
            Self::IdentityDeletion { .. } => vec![],
            Self::Reauthorization { iris_pair, .. }
            | Self::ResetCheck { iris_pair, .. }
            | Self::ResetUpdate { iris_pair, .. }
            | Self::Uniqueness { iris_pair, .. } => match iris_pair {
                Some(iris_pair) => vec![iris_pair.left().index(), iris_pair.right().index()],
                None => vec![],
            },
        }
    }

    /// Returns true if a system response is deemed to be correlated with this system request.
    pub fn is_correlation(&self, response: &ResponsePayload) -> bool {
        match (self, response) {
            (Self::IdentityDeletion { parent, .. }, ResponsePayload::IdentityDeletion(result)) => {
                *parent == result.serial_id
            }
            (Self::Reauthorization { reauth_id, .. }, ResponsePayload::Reauthorization(result)) => {
                result.reauth_id == reauth_id.to_string()
            }
            (Self::ResetCheck { reset_id, .. }, ResponsePayload::ResetCheck(result)) => {
                result.reset_id == reset_id.to_string()
            }
            (Self::ResetUpdate { reset_id, .. }, ResponsePayload::ResetUpdate(result)) => {
                result.reset_id == reset_id.to_string()
            }
            (Self::Uniqueness { signup_id, .. }, ResponsePayload::Uniqueness(result)) => {
                result.signup_id == signup_id.to_string()
            }
            _ => false,
        }
    }

    /// Returns true if request has been enqueued for system processing.
    pub fn is_enqueued(&self) -> bool {
        matches!(self.info().status(), RequestStatus::Enqueued)
    }

    /// Returns true if shares have been uploaded and the request is ready to enqueue.
    /// Since the parent serial ID is always known at Request creation time, no extra check needed.
    pub fn is_enqueueable(&self) -> bool {
        matches!(self.info().status(), RequestStatus::SharesUploaded)
    }

    pub fn label(&self) -> &Option<String> {
        self.info().label()
    }

    pub fn has_error_response(&self) -> bool {
        self.info().has_error_response()
    }

    /// Records a node response. Returns true if all parties have now responded (request is Complete).
    pub fn record_response(&mut self, response: &ResponsePayload) -> bool {
        tracing::info!("{} :: response -> Node-{}", &self, response.node_id());
        let is_complete = self.info_mut().record_response(response);
        if is_complete {
            self.set_status(RequestStatus::Complete);
        }
        is_complete
    }

    /// Updates request status.
    pub fn set_status(&mut self, new_state: RequestStatus) {
        tracing::info!("{} :: {}", &self, new_state);
        self.info_mut().set_status(new_state);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion { .. } => write!(f, "{}.IdentityDeletion", self.info()),
            Self::Reauthorization { .. } => write!(f, "{}.Reauthorization", self.info()),
            Self::ResetCheck { .. } => write!(f, "{}.ResetCheck", self.info()),
            Self::ResetUpdate { .. } => write!(f, "{}.ResetUpdate", self.info()),
            Self::Uniqueness { .. } => write!(f, "{}.Uniqueness", self.info()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Request;

    impl Request {
        pub fn is_identity_deletion(&self) -> bool {
            matches!(self, Self::IdentityDeletion { .. })
        }

        pub fn is_reauthorization(&self) -> bool {
            matches!(self, Self::Reauthorization { .. })
        }

        pub fn is_reset_check(&self) -> bool {
            matches!(self, Self::ResetCheck { .. })
        }

        pub fn is_reset_update(&self) -> bool {
            matches!(self, Self::ResetUpdate { .. })
        }

        pub fn is_uniqueness(&self) -> bool {
            matches!(self, Self::Uniqueness { .. })
        }
    }
}
