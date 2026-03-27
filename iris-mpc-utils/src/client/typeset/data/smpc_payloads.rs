use serde::{Deserialize, Serialize};

use iris_mpc_common::helpers::{smpc_request, smpc_response};

use crate::{aws::types::SqsMessageInfo, client::typeset::Request};

// both iris-mpc-common and iris-mpc-utils need this enumeration. But
// iris_mpc_utils::RequestPayload also needs to implement From<Request>, which
// has to exist in the module that the enum was defined in).
// Also, iris-mpc-common probably shouldn't import anything from iris-mpc-utils.
// To deal with this, define the duplicate type here, in iris-mpc-utils
#[allow(clippy::large_enum_variant)]
pub enum RequestPayload {
    Uniqueness(smpc_request::UniquenessRequest),
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::IdentityMatchCheckRequest),
    RecoveryCheck(smpc_request::IdentityMatchCheckRequest),
    ResetUpdate(smpc_request::IdentityUpdateRequest),
    RecoveryUpdate(smpc_request::IdentityUpdateRequest),
}

impl RequestPayload {
    pub fn to_smpc_request(self) -> smpc_request::RequestPayload {
        match self {
            Self::Uniqueness(x) => smpc_request::RequestPayload::Uniqueness(x),
            Self::IdentityDeletion(x) => smpc_request::RequestPayload::IdentityDeletion(x),
            Self::Reauthorization(x) => smpc_request::RequestPayload::Reauthorization(x),
            Self::ResetCheck(x) => smpc_request::RequestPayload::ResetCheck(x),
            Self::RecoveryCheck(x) => smpc_request::RequestPayload::RecoveryCheck(x),
            Self::ResetUpdate(x) => smpc_request::RequestPayload::ResetUpdate(x),
            Self::RecoveryUpdate(x) => smpc_request::RequestPayload::RecoveryUpdate(x),
        }
    }
}

/// Enumeration over response messages dequeued from system egress queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponsePayload {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::IdentityMatchCheckResult),
    RecoveryCheck(smpc_response::IdentityMatchCheckResult),
    ResetUpdate(smpc_response::IdentityUpdateAckResult),
    RecoveryUpdate(smpc_response::IdentityUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}

impl ResponsePayload {
    pub(super) fn node_id(&self) -> usize {
        match self {
            Self::IdentityDeletion(result) => result.node_id,
            Self::Reauthorization(result) => result.node_id,
            Self::ResetCheck(result) => result.node_id,
            Self::RecoveryCheck(result) => result.node_id,
            Self::ResetUpdate(result) => result.node_id,
            Self::RecoveryUpdate(result) => result.node_id,
            Self::Uniqueness(result) => result.node_id,
        }
    }

    /// Returns true if the response indicates a processing error.
    pub fn is_error(&self) -> bool {
        match self {
            Self::IdentityDeletion(result) => !result.success,
            Self::Reauthorization(result) => result.error.unwrap_or(false),
            Self::RecoveryCheck(result) => result.error.unwrap_or(false),
            Self::ResetCheck(result) => result.error.unwrap_or(false),
            Self::ResetUpdate(result) => result.error.unwrap_or(false),
            Self::RecoveryUpdate(result) => result.error.unwrap_or(false),
            Self::Uniqueness(result) => result.error.unwrap_or(false),
        }
    }

    /// Returns a log tag containing the response type, node ID, and operation identifier.
    pub fn log_tag(&self) -> String {
        match self {
            Self::IdentityDeletion(result) => {
                format!(
                    "IdentityDeletion | node={} | serial={} | success={}",
                    result.node_id, result.serial_id, result.success
                )
            }
            Self::Reauthorization(result) => {
                format!(
                    "Reauthorization | node={} | serial_id={} | success={} | reauth_id={:.16}",
                    result.node_id, result.serial_id, result.success, result.reauth_id
                )
            }
            Self::RecoveryCheck(result) => {
                format!(
                    "RecoveryCheck | node={} | request_id={:.16}",
                    result.node_id, result.request_id
                )
            }
            Self::ResetCheck(result) => {
                format!(
                    "ResetCheck | node={} | request_id={:.16}",
                    result.node_id, result.request_id
                )
            }
            Self::ResetUpdate(result) => {
                format!(
                    "ResetUpdate | node={} | serial_id={} | reset_id={:.16}",
                    result.node_id, result.serial_id, result.request_id
                )
            }
            Self::RecoveryUpdate(result) => {
                format!(
                    "RecoveryUpdate | node={} | serial_id={} | recovery_id={:.16}",
                    result.node_id, result.serial_id, result.request_id
                )
            }
            Self::Uniqueness(result) => {
                format!(
                    "Uniqueness | node={} |serial_id={} | is_match={} | signup_id={:.16}",
                    result.node_id,
                    result
                        .get_serial_id()
                        .map(|id| format!("{id}"))
                        .unwrap_or(String::from("_")),
                    result.is_match,
                    result.signup_id
                )
            }
        }
    }

    /// Returns the error reason string if one is available.
    pub fn error_reason(&self) -> Option<&str> {
        match self {
            Self::IdentityDeletion(_) => None,
            Self::Reauthorization(result) => result.error_reason.as_deref(),
            Self::RecoveryCheck(result) => result.error_reason.as_deref(),
            Self::ResetCheck(result) => result.error_reason.as_deref(),
            Self::ResetUpdate(result) => result.error_reason.as_deref(),
            Self::RecoveryUpdate(result) => result.error_reason.as_deref(),
            Self::Uniqueness(result) => result.error_reason.as_deref(),
        }
    }

    /// Validates the response against expected field values.
    pub fn matches_expected(&self, expected: &serde_json::Value) -> Result<(), Vec<String>> {
        match self {
            Self::IdentityDeletion(result) => result.matches_expected(expected),
            Self::Reauthorization(result) => result.matches_expected(expected),
            Self::RecoveryCheck(result) => result.matches_expected(expected),
            Self::ResetCheck(result) => result.matches_expected(expected),
            Self::ResetUpdate(result) => result.matches_expected(expected),
            Self::RecoveryUpdate(result) => result.matches_expected(expected),
            Self::Uniqueness(result) => result.matches_expected(expected),
        }
    }
}

impl From<&Request> for RequestPayload {
    fn from(request: &Request) -> Self {
        match request {
            Request::IdentityDeletion { parent, .. } => {
                Self::IdentityDeletion(smpc_request::IdentityDeletionRequest { serial_id: *parent })
            }
            Request::Reauthorization {
                reauth_id, parent, ..
            } => Self::Reauthorization(smpc_request::ReAuthRequest {
                reauth_id: reauth_id.to_string(),
                s3_key: reauth_id.to_string(),
                serial_id: *parent,
                skip_persistence: None,
                use_or_rule: false,
            }),
            Request::RecoveryCheck { request_id, .. } => {
                Self::RecoveryCheck(smpc_request::IdentityMatchCheckRequest {
                    request_id: request_id.to_string(),
                    s3_key: request_id.to_string(),
                })
            }
            Request::ResetCheck { reset_id, .. } => {
                Self::ResetCheck(smpc_request::IdentityMatchCheckRequest {
                    request_id: reset_id.to_string(),
                    s3_key: reset_id.to_string(),
                })
            }
            Request::ResetUpdate {
                reset_id, parent, ..
            } => Self::ResetUpdate(smpc_request::IdentityUpdateRequest {
                request_id: reset_id.to_string(),
                s3_key: reset_id.to_string(),
                serial_id: *parent,
            }),
            Request::RecoveryUpdate {
                recovery_id,
                parent,
                ..
            } => Self::RecoveryUpdate(smpc_request::IdentityUpdateRequest {
                request_id: recovery_id.to_string(),
                s3_key: recovery_id.to_string(),
                serial_id: *parent,
            }),
            Request::Uniqueness { signup_id, .. } => {
                Self::Uniqueness(smpc_request::UniquenessRequest {
                    s3_key: signup_id.to_string(),
                    signup_id: signup_id.to_string(),
                    or_rule_serial_ids: None,
                    skip_persistence: None,
                    full_face_mirror_attacks_detection_enabled: Some(true),
                    disable_anonymized_stats: None,
                })
            }
        }
    }
}

impl TryFrom<&SqsMessageInfo> for ResponsePayload {
    type Error = serde_json::Error;
    fn try_from(msg: &SqsMessageInfo) -> Result<Self, Self::Error> {
        let body = msg.body();
        let kind = msg.kind();

        macro_rules! parse_response {
            ($variant:ident) => {
                serde_json::from_str(body).map(|x| ResponsePayload::$variant(x))
            };
        }

        match kind {
            iris_mpc_common::helpers::smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                parse_response!(IdentityDeletion)
            }
            iris_mpc_common::helpers::smpc_request::REAUTH_MESSAGE_TYPE => {
                parse_response!(Reauthorization)
            }
            iris_mpc_common::helpers::smpc_request::RECOVERY_CHECK_MESSAGE_TYPE => {
                parse_response!(RecoveryCheck)
            }
            iris_mpc_common::helpers::smpc_request::RESET_CHECK_MESSAGE_TYPE => {
                parse_response!(ResetCheck)
            }
            iris_mpc_common::helpers::smpc_request::RESET_UPDATE_MESSAGE_TYPE => {
                parse_response!(ResetUpdate)
            }
            iris_mpc_common::helpers::smpc_request::RECOVERY_UPDATE_MESSAGE_TYPE => {
                parse_response!(RecoveryUpdate)
            }
            iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE => {
                parse_response!(Uniqueness)
            }
            _ => Err(serde::de::Error::custom(format!(
                "Unsupported system response type: {kind}"
            ))),
        }
    }
}
