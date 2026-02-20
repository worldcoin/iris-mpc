use iris_mpc_common::helpers::smpc_request;

/// Typed representation of a batch request kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    IdentityDeletion,
    Reauth,
    ResetCheck,
    ResetUpdate,
    Uniqueness,
}

impl BatchKind {
    /// Parses a batch kind from its SMPC message type string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => Some(Self::IdentityDeletion),
            smpc_request::REAUTH_MESSAGE_TYPE => Some(Self::Reauth),
            smpc_request::RESET_CHECK_MESSAGE_TYPE => Some(Self::ResetCheck),
            smpc_request::RESET_UPDATE_MESSAGE_TYPE => Some(Self::ResetUpdate),
            smpc_request::UNIQUENESS_MESSAGE_TYPE => Some(Self::Uniqueness),
            _ => None,
        }
    }

    /// Returns true if this batch kind requires a parent uniqueness request.
    pub fn requires_parent(&self) -> bool {
        matches!(
            self,
            Self::IdentityDeletion | Self::Reauth | Self::ResetUpdate
        )
    }
}
