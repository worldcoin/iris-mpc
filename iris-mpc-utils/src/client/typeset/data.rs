use std::fmt;

use uuid;

use iris_mpc_common::{
    helpers::{
        smpc_request::{
            self, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
            RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
        },
        smpc_response,
    },
    IrisSerialId,
};

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
}

impl RequestInfo {
    pub fn is_correlated(&self) -> bool {
        self.correlation_set.iter().all(|c| c.is_some())
    }

    fn set_correlation(&mut self, response: &ResponseBody) {
        self.correlation_set[response.node_id()] = Some(response.to_owned());
        tracing::info!("{} :: Correlated -> Node-{}", &self, response.node_id());
        if self.is_correlated() {
            self.set_status(RequestStatus::Correlated);
            tracing::info!("{} :: Correlated", &self);
        }
    }

    fn set_status(&mut self, new_state: RequestStatus) {
        self.status = new_state;
        tracing::info!("{} :: State -> {:?}", &self, self.status);
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request {}.{}", self.batch_idx, self.batch_item_idx)
    }
}

impl From<&RequestBatch> for RequestInfo {
    fn from(batch: &RequestBatch) -> Self {
        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.next_item_idx(),
            correlation_set: [const { None }; N_PARTIES],
            status: RequestStatus::default(),
        }
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
        matches!(self.status(), RequestStatus::New)
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

    pub fn set_status_enqueued(&mut self) {
        self.info_mut().set_status(RequestStatus::Enqueued);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.info(), self.label())
    }
}

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Debug)]
pub struct RequestBatch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Requests in batch.
    requests: Vec<Request>,
}

impl RequestBatch {
    pub fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub fn requests(&self) -> &[Request] {
        self.requests.as_slice()
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Request> {
        &mut self.requests
    }

    pub fn new(batch_idx: usize, batch_size: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::with_capacity(batch_size * 2),
        }
    }

    pub fn enqueued_mut(&mut self) -> impl Iterator<Item = &mut Request> {
        self.requests_mut().iter_mut().filter(|r| r.is_enqueued())
    }

    pub fn has_enqueued_items(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueued())
    }

    pub fn is_enqueueable(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueueable())
    }

    pub fn next_item_idx(&self) -> usize {
        &self.requests.len() + 1
    }

    pub fn push(&mut self, request: Request) {
        self.requests_mut().push(request);
    }
}

impl fmt::Display for RequestBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Batch:{:03}", self.batch_idx)
    }
}

/// Encapsulates inputs used to derive the kind of request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchKind {
    /// All requests are of same type.
    Simple(&'static str),
}

impl fmt::Display for RequestBatchKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Simple(kind) => write!(f, "{}", kind),
        }
    }
}

impl From<&String> for RequestBatchKind {
    fn from(option: &String) -> Self {
        Self::Simple(match option.as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => IDENTITY_DELETION_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE => REAUTH_MESSAGE_TYPE,
            RESET_CHECK_MESSAGE_TYPE => RESET_CHECK_MESSAGE_TYPE,
            RESET_UPDATE_MESSAGE_TYPE => RESET_UPDATE_MESSAGE_TYPE,
            UNIQUENESS_MESSAGE_TYPE => UNIQUENESS_MESSAGE_TYPE,
            _ => panic!("Unsupported request batch kind"),
        })
    }
}

/// Encapsulates inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchSize {
    /// Batch size is static.
    Static(usize),
}

impl fmt::Display for RequestBatchSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(size) => write!(f, "{}", size),
        }
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

/// Enumeration over request processing states.
#[derive(Debug, Default, Clone)]
pub enum RequestStatus {
    // Associated responses have been correlated.
    Correlated,
    // Enqueued upon system ingress queue.
    Enqueued,
    // Processed by system but an error occurred.
    Error,
    // Newly generated by client.
    #[default]
    New,
    // Processed by system sucessfully.
    Success,
}

pub struct RequestFactory {}

impl RequestFactory {
    pub fn new_identity_deletion(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::IdentityDeletion {
                info: RequestInfo::from(batch),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::Reauthorization {
                info: RequestInfo::from(batch),
                reauth_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Request {
        Request::ResetCheck {
            info: RequestInfo::from(batch),
            reset_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: &Request) -> Request {
        match parent {
            Request::Uniqueness { signup_id, .. } => Request::ResetUpdate {
                info: RequestInfo::from(batch),
                reset_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Request {
        Request::Uniqueness {
            info: RequestInfo::from(batch),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}

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

    pub fn serial_id(&self) -> u32 {
        match self {
            Self::IdentityDeletion(result) => result.serial_id,
            Self::Reauthorization(result) => result.serial_id,
            Self::ResetCheck(_) => panic!("ResetCheck has no associated serial-id"),
            Self::ResetUpdate(result) => result.serial_id,
            Self::Uniqueness(result) => result.serial_id.unwrap(),
        }
    }
}
