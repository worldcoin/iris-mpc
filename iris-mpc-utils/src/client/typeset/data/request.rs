use std::{fmt, time::Instant};

use uuid;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{request_batch::RequestBatch, request_info::RequestInfo, response::ResponseBody};

/// Encapsulates data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub enum Request {
    IdentityDeletion {
        // Standard request information.
        info: RequestInfo,
        // Iris serial identifier.
        serial_id: Option<IrisSerialId>,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
    },
    Reauthorization {
        // Standard request information.
        info: RequestInfo,
        // Operation identifier.
        reauth_id: uuid::Uuid,
        // Iris serial identifier.
        serial_id: Option<IrisSerialId>,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
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
        // Operation identifier.
        reset_id: uuid::Uuid,
        // Iris serial identifier.
        serial_id: Option<IrisSerialId>,
        // Iris sign-up identifier.
        signup_id: uuid::Uuid,
    },
    Uniqueness {
        // Standard request information.
        info: RequestInfo,
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

    pub fn info_mut(&mut self) -> &mut RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::IdentityDeletion { .. } => "IdentityDeletion",
            Self::Reauthorization { .. } => "Reauthorization",
            Self::ResetCheck { .. } => "ResetCheck",
            Self::ResetUpdate { .. } => "ResetUpdate",
            Self::Uniqueness { .. } => "Uniqueness",
        }
    }

    /// Returns identifier to be assigned to associated iris shares.
    pub fn shares_id(&self) -> Option<&uuid::Uuid> {
        match self {
            Self::IdentityDeletion { .. } => None,
            Self::Reauthorization { reauth_id, .. } => Some(reauth_id),
            Self::ResetCheck { reset_id, .. } => Some(reset_id),
            Self::ResetUpdate { reset_id, .. } => Some(reset_id),
            Self::Uniqueness { signup_id, .. } => Some(signup_id),
        }
    }

    pub fn status(&self) -> &RequestStatus {
        self.info().status()
    }

    /// Returns true if a system response is deemed to be correlated with this system request.
    pub fn is_correlated(&self, response: &ResponseBody) -> bool {
        match (self, response) {
            (Self::IdentityDeletion { serial_id, .. }, ResponseBody::IdentityDeletion(result)) => {
                result.serial_id == serial_id.unwrap()
            }
            (Self::Reauthorization { reauth_id, .. }, ResponseBody::Reauthorization(result)) => {
                result.reauth_id == reauth_id.to_string()
            }
            (Self::ResetCheck { reset_id, .. }, ResponseBody::ResetCheck(result)) => {
                result.reset_id == reset_id.to_string()
            }
            (Self::ResetUpdate { reset_id, .. }, ResponseBody::ResetUpdate(result)) => {
                result.reset_id == reset_id.to_string()
            }
            (Self::Uniqueness { signup_id, .. }, ResponseBody::Uniqueness(result)) => {
                result.signup_id == signup_id.to_string()
            }
            _ => false,
        }
    }

    /// Returns true if request has been enqueued for system processing.
    pub fn is_enqueued(&self) -> bool {
        matches!(self.status(), RequestStatus::Enqueued(_))
    }

    /// Returns true if generated and not awaiting data returned from a parent request.
    pub fn is_enqueueable(&self) -> bool {
        matches!(self.status(), RequestStatus::SharesUploaded(_))
            && match self {
                Self::IdentityDeletion { serial_id, .. } => serial_id.is_some(),
                Self::Reauthorization { serial_id, .. } => serial_id.is_some(),
                Self::ResetUpdate { serial_id, .. } => serial_id.is_some(),
                _ => true,
            }
    }

    /// Sets a correlated response (if correlated).
    pub fn maybe_set_correlation(&mut self, response: &ResponseBody) -> bool {
        let is_correlated = self.is_correlated(response);
        if is_correlated {
            self.info_mut().set_correlation(response);
        }
        is_correlated
    }

    pub fn set_status(&mut self, new_state: RequestStatus) {
        self.info_mut().set_status(new_state);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.info(), self.label())
    }
}

/// Enumeration over request message body for dispatch to system ingress queue.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestBody {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}

pub struct RequestFactory {}

impl RequestFactory {
    pub fn new_identity_deletion(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::IdentityDeletion {
                info: RequestInfo::new(batch),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::Reauthorization {
                info: RequestInfo::new(batch),
                reauth_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Request {
        Request::ResetCheck {
            info: RequestInfo::new(batch),
            reset_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::ResetUpdate {
                info: RequestInfo::new(batch),
                reset_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Request {
        Request::Uniqueness {
            info: RequestInfo::new(batch),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}

/// Enumeration over request processing states plus time of state change for statisitical purposes.
#[derive(Debug, Clone)]
pub enum RequestStatus {
    // Associated responses have been correlated.
    Correlated(Instant),
    // Enqueued upon system ingress queue.
    Enqueued(Instant),
    // Newly generated by client.
    New(Instant),
    // Iris shares uploaded in advance of enqueuement.
    SharesUploaded(Instant),
}

impl RequestStatus {
    pub fn new_correlated() -> Self {
        Self::Correlated(Instant::now())
    }

    pub fn new_enqueued() -> Self {
        Self::Enqueued(Instant::now())
    }

    pub fn new_shares_uploaded() -> Self {
        Self::SharesUploaded(Instant::now())
    }
}

impl Default for RequestStatus {
    fn default() -> Self {
        Self::New(Instant::now())
    }
}
