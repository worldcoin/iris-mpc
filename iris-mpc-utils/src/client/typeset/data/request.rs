use std::{
    fmt,
    time::{Duration, Instant},
};

use uuid;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{request_batch::RequestBatch, response::ResponseBody};
use crate::constants::N_PARTIES;

/// Encapsulates common data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub struct RequestInfo {
    /// Associated request batch ordinal identifier.
    batch_idx: usize,

    /// Associated request batch item ordinal identifier.
    batch_item_idx: usize,

    /// Correlated system responses returned by MPC nodes.
    correlation_set: [Option<ResponseBody>; N_PARTIES],

    /// Current processing state.
    status: RequestStatus,

    /// Instant at point of instantiation.
    time_new: Instant,

    /// Duration from instantiation until correlated.
    time_to_correlation: Option<Duration>,

    /// Duration from instantiation until enqueued.
    time_to_enqueuement: Option<Duration>,
}

impl RequestInfo {
    fn new(batch: &RequestBatch) -> Self {
        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.next_item_idx(),
            correlation_set: [const { None }; N_PARTIES],
            status: RequestStatus::default(),
            time_to_correlation: None,
            time_to_enqueuement: None,
            time_new: batch.time_new().clone(),
        }
    }

    pub fn is_correlated(&self) -> bool {
        self.correlation_set.iter().all(|c| c.is_some())
    }

    fn set_correlation(&mut self, response: &ResponseBody) {
        self.correlation_set[response.node_id()] = Some(response.to_owned());
        tracing::info!("{} :: Correlated -> Node-{}", &self, response.node_id());
        if self.is_correlated() {
            self.set_status(RequestStatus::Correlated);
        }
    }

    fn set_status(&mut self, new_state: RequestStatus) {
        self.status = new_state;
        match self.status {
            RequestStatus::Enqueued => self.time_to_enqueuement = Some(self.time_new.elapsed()),
            RequestStatus::Correlated => self.time_to_correlation = Some(self.time_new.elapsed()),
            _ => {}
        }
        tracing::info!("{} :: State -> {:?}", &self, self.status);
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request {}.{}", self.batch_idx, self.batch_item_idx)
    }
}

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
        &self.info().status
    }

    /// Returns true if a system response is deemed to be correlated with this system request.
    fn is_correlated_response(&self, response: &ResponseBody) -> bool {
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
        matches!(self.status(), RequestStatus::Enqueued)
    }

    /// Returns true if generated and not awaiting data returned from a parent request.
    pub fn is_enqueueable(&self) -> bool {
        matches!(self.status(), RequestStatus::DataUploaded)
            && match self {
                Self::IdentityDeletion { serial_id, .. } => serial_id.is_some(),
                Self::Reauthorization { serial_id, .. } => serial_id.is_some(),
                Self::ResetUpdate { serial_id, .. } => serial_id.is_some(),
                _ => true,
            }
    }

    /// Sets a correlated response (if correlated).
    pub fn maybe_set_correlation(&mut self, response: &ResponseBody) -> bool {
        let is_correlated = self.is_correlated_response(response);
        if is_correlated {
            self.info_mut().set_correlation(response);
        }
        is_correlated
    }

    pub fn set_status_data_uploaded(&mut self) {
        self.info_mut().set_status(RequestStatus::DataUploaded);
    }

    pub fn set_status_enqueued(&mut self) {
        self.info_mut().set_status(RequestStatus::Enqueued);
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

/// Enumeration over request processing states.
#[derive(Debug, Default, Clone)]
pub enum RequestStatus {
    // Associated responses have been correlated.
    Correlated,
    // Data uploaded in advance of enqueuement.
    DataUploaded,
    // Enqueued upon system ingress queue.
    Enqueued,
    // Newly generated by client.
    #[default]
    New,
}
