use std::fmt;

use serde::{Deserialize, Serialize};
use uuid;

use iris_mpc_common::IrisSerialId;

use super::{IrisPairDescriptor, RequestInfo};

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
    RecoveryCheck {
        // Standard request information.
        info: RequestInfo,
        // Associated Iris pair descriptor ... used to build deterministic graphs.
        iris_pair: Option<IrisPairDescriptor>,
        // Operation identifier.
        request_id: uuid::Uuid,
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
            | Self::RecoveryCheck { info, .. }
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
            Request::RecoveryCheck {
                request_id,
                iris_pair,
                ..
            } => Some((*request_id, *iris_pair)),
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

    /// Returns a log tag containing the request type, label (if set), operation UUID, and iris serial ID (if present).
    pub fn log_tag(&self) -> String {
        let kind = match self {
            Self::IdentityDeletion { .. } => "IdentityDeletion",
            Self::Reauthorization { .. } => "Reauthorization",
            Self::RecoveryCheck { .. } => "RecoveryCheck",
            Self::ResetCheck { .. } => "ResetCheck",
            Self::ResetUpdate { .. } => "ResetUpdate",
            Self::Uniqueness { .. } => "Uniqueness",
        };
        let label = self.info().label();
        let serial_id: Option<IrisSerialId> = match self {
            Self::IdentityDeletion { parent, .. }
            | Self::Reauthorization { parent, .. }
            | Self::ResetUpdate { parent, .. } => Some(*parent),
            Self::RecoveryCheck { .. } | Self::ResetCheck { .. } | Self::Uniqueness { .. } => None,
        };
        let op_id: Option<(&str, uuid::Uuid)> = match self {
            Self::Uniqueness { signup_id, .. } => Some(("signup_id", *signup_id)),
            Self::Reauthorization { reauth_id, .. } => Some(("reauth_id", *reauth_id)),
            Self::RecoveryCheck { request_id, .. } => Some(("request_id", *request_id)),
            Self::ResetCheck { reset_id, .. } | Self::ResetUpdate { reset_id, .. } => {
                Some(("reset_id", *reset_id))
            }
            Self::IdentityDeletion { .. } => None,
        };

        let mut parts = vec![kind.to_string()];
        if let Some(lbl) = label {
            parts.push(format!("{:.16}", lbl));
        }
        if let Some(sid) = serial_id {
            parts.push(format!("serial={}", sid));
        }
        if let Some((key, id)) = op_id {
            parts.push(format!("{}={:.16}", key, id.to_string()));
        }
        parts.join(" | ")
    }

    pub fn has_error_response(&self) -> bool {
        self.info().has_error_response()
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion { .. } => write!(f, "{}.IdentityDeletion", self.info()),
            Self::Reauthorization { .. } => write!(f, "{}.Reauthorization", self.info()),
            Self::RecoveryCheck { .. } => write!(f, "{}.RecoveryCheck", self.info()),
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

        pub fn is_recovery_check(&self) -> bool {
            matches!(self, Self::RecoveryCheck { .. })
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
