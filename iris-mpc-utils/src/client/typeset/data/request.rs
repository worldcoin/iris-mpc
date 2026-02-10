use std::fmt;

use serde::{Deserialize, Serialize};
use uuid;

use super::{
    IrisPairDescriptor, RequestInfo, RequestStatus, ResponsePayload, UniquenessRequestDescriptor,
};

/// Encapsulates data pertinent to a system processing request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    IdentityDeletion {
        // Standard request information.
        info: RequestInfo,
        // Weak reference to associated uniqueness request.
        uniqueness_ref: UniquenessRequestDescriptor,
    },
    Reauthorization {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair_ref: Option<IrisPairDescriptor>,
        // Operation identifier.
        reauth_id: uuid::Uuid,
        // Weak reference to associated uniqueness request.
        uniqueness_ref: UniquenessRequestDescriptor,
    },
    ResetCheck {
        // Standard request information.
        info: RequestInfo,
        // Operation identifier.
        reset_id: uuid::Uuid,
    },
    ResetUpdate {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair_ref: Option<IrisPairDescriptor>,
        // Operation identifier.
        reset_id: uuid::Uuid,
        // Weak reference to associated uniqueness request.
        uniqueness_ref: UniquenessRequestDescriptor,
    },
    Uniqueness {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair_ref: Option<IrisPairDescriptor>,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
    },
}

impl Request {
    fn info(&self) -> &RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
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

    /// Returns true if deemed a child of previous system request.
    pub(super) fn is_child(&self, parent: &Self) -> bool {
        // A child of a uniqueness request can be derived by comparing signup identifiers.
        match self {
            Self::IdentityDeletion { uniqueness_ref, .. }
            | Self::Reauthorization { uniqueness_ref, .. }
            | Self::ResetUpdate { uniqueness_ref, .. } => {
                matches!(
                    (uniqueness_ref, parent),
                    (UniquenessRequestDescriptor::SignupId(uniqueness_signup_id), Self::Uniqueness { signup_id, .. })
                    if signup_id == uniqueness_signup_id
                )
            }
            _ => false,
        }
    }

    /// Returns true if a system response is deemed to be correlated with this system request.
    pub(super) fn is_correlation(&self, response: &ResponsePayload) -> bool {
        match (self, response) {
            (
                Self::IdentityDeletion { uniqueness_ref, .. },
                ResponsePayload::IdentityDeletion(result),
            ) => matches!(
                uniqueness_ref,
                UniquenessRequestDescriptor::IrisSerialId(serial_id) if *serial_id == result.serial_id
            ),
            (Self::Reauthorization { reauth_id, .. }, ResponsePayload::Reauthorization(result)) => {
                reauth_id.to_string() == result.reauth_id
            }
            (Self::ResetCheck { reset_id, .. }, ResponsePayload::ResetCheck(result)) => {
                reset_id.to_string() == result.reset_id
            }
            (Self::ResetUpdate { reset_id, .. }, ResponsePayload::ResetUpdate(result)) => {
                reset_id.to_string() == result.reset_id
            }
            (Self::Uniqueness { signup_id, .. }, ResponsePayload::Uniqueness(result)) => {
                signup_id.to_string() == result.signup_id
            }
            _ => false,
        }
    }

    /// Returns true if request has been enqueued for system processing.
    pub(super) fn is_enqueued(&self) -> bool {
        matches!(self.info().status(), RequestStatus::Enqueued)
    }

    /// Returns true if generated and not awaiting data returned from a parent request.
    pub(crate) fn is_enqueueable(&self) -> bool {
        matches!(self.info().status(), RequestStatus::SharesUploaded)
            && match self {
                Self::IdentityDeletion { uniqueness_ref, .. } => {
                    matches!(uniqueness_ref, UniquenessRequestDescriptor::IrisSerialId(_))
                }
                Self::Reauthorization { uniqueness_ref, .. } => {
                    matches!(uniqueness_ref, UniquenessRequestDescriptor::IrisSerialId(_))
                }
                Self::ResetUpdate { uniqueness_ref, .. } => {
                    matches!(uniqueness_ref, UniquenessRequestDescriptor::IrisSerialId(_))
                }
                _ => true,
            }
    }

    /// Sets correlated response and maybe sets request state.
    pub(crate) fn set_correlation(&mut self, response: &ResponsePayload) -> Option<()> {
        tracing::info!("{} :: Correlated -> Node-{}", &self, response.node_id());
        self.info_mut().set_correlation(response);
        self.info().is_fully_correlated().then(|| {
            self.set_status(RequestStatus::Correlated);
        })
    }

    /// Updates request status.
    pub(crate) fn set_status(&mut self, new_state: RequestStatus) {
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
