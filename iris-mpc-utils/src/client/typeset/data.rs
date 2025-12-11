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

/// Encapsulates common data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub struct RequestInfo {
    /// Associated batch ordinal identifier.
    batch_idx: usize,

    /// Associated batch item ordinal identifier.
    batch_item_idx: usize,

    /// Associated universally unique identifier.
    request_id: uuid::Uuid,

    /// Associated set of correlated responses.
    correlation_set: ResponseCorrelationSet,

    /// Associated processing state.
    status: RequestStatus,
}

impl RequestInfo {
    pub fn correlation_set_mut(&mut self) -> &mut ResponseCorrelationSet {
        &mut self.correlation_set
    }

    pub fn new(batch_idx: usize, batch_item_idx: usize) -> Self {
        Self {
            batch_idx,
            batch_item_idx,
            request_id: uuid::Uuid::new_v4(),
            correlation_set: ResponseCorrelationSet::default(),
            status: RequestStatus::default(),
        }
    }
}

/// Encapsulates data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub enum Request {
    IdentityDeletion {
        info: RequestInfo,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
    },
    Reauthorization {
        info: RequestInfo,
        reauth_id: uuid::Uuid,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
    },
    ResetCheck {
        info: RequestInfo,
        reset_check_id: uuid::Uuid,
    },
    ResetUpdate {
        info: RequestInfo,
        reset_update_id: uuid::Uuid,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
    },
    Uniqueness {
        info: RequestInfo,
        signup_id: uuid::Uuid,
    },
}

impl Request {
    pub fn batch_idx(&self) -> usize {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info.batch_idx,
        }
    }

    pub fn batch_item_idx(&self) -> usize {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => info.batch_item_idx,
        }
    }

    pub fn info(&self) -> &RequestInfo {
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

    pub fn request_id(&self) -> &uuid::Uuid {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => &info.request_id,
        }
    }

    pub fn signup_id(&self) -> Option<&uuid::Uuid> {
        match self {
            Self::IdentityDeletion { signup_id, .. }
            | Self::Reauthorization { signup_id, .. }
            | Self::ResetUpdate { signup_id, .. }
            | Self::Uniqueness { signup_id, .. } => Some(signup_id),
            Self::ResetCheck { .. } => None,
        }
    }

    pub fn status(&self) -> &RequestStatus {
        match self {
            Self::IdentityDeletion { info, .. }
            | Self::Reauthorization { info, .. }
            | Self::ResetCheck { info, .. }
            | Self::ResetUpdate { info, .. }
            | Self::Uniqueness { info, .. } => &info.status,
        }
    }

    pub fn new_identity_deletion(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::IdentityDeletion {
                info: RequestInfo::new(batch.batch_idx, batch.next_item_idx()),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::Reauthorization {
                info: RequestInfo::new(batch.batch_idx, batch.next_item_idx()),
                reauth_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Self {
        Self::ResetCheck {
            info: RequestInfo::new(batch.batch_idx, batch.next_item_idx()),
            reset_check_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::ResetUpdate {
                info: RequestInfo::new(batch.batch_idx, batch.next_item_idx()),
                reset_update_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
            },
            _ => panic!("Invalid request parent"),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Self {
        Self::Uniqueness {
            info: RequestInfo::new(batch.batch_idx, batch.next_item_idx()),
            signup_id: uuid::Uuid::new_v4(),
        }
    }

    /// Returns true if request system processing is complete.
    pub fn is_complete(&self) -> bool {
        matches!(self.status(), RequestStatus::Complete)
    }

    /// Returns true if request has been enqueued for system processing.
    pub fn is_enqueued(&self) -> bool {
        matches!(self.status(), RequestStatus::Enqueued)
    }

    /// Returns true if request system processing resulted in an error.
    pub fn is_error(&self) -> bool {
        matches!(self.status(), RequestStatus::Error)
    }

    /// Returns true if the response is deemed to be correlated with the request.
    pub fn is_correlated(&self, response: &Response) -> bool {
        match response {
            Response::IdentityDeletion(result) => match self {
                Self::IdentityDeletion { serial_id, .. } => result.serial_id == serial_id.unwrap(),
                _ => false,
            },
            Response::Reauthorization(result) => match self {
                Self::Reauthorization { reauth_id, .. } => {
                    result.reauth_id == reauth_id.to_string()
                }
                _ => false,
            },
            Response::ResetCheck(result) => match self {
                Self::ResetCheck { reset_check_id, .. } => {
                    result.reset_id == reset_check_id.to_string()
                }
                _ => false,
            },
            Response::ResetUpdate(result) => match self {
                Self::ResetUpdate {
                    reset_update_id, ..
                } => result.reset_id == reset_update_id.to_string(),
                _ => false,
            },
            Response::Uniqueness(result) => match self {
                Self::Uniqueness { signup_id, .. } => result.signup_id == signup_id.to_string(),
                _ => false,
            },
        }
    }

    /// Returns true if generated and not awaiting data returned from a parent request.
    pub fn is_enqueueable(&self) -> bool {
        matches!(self.status(), RequestStatus::Generated)
            && match self {
                Self::IdentityDeletion { serial_id, .. } => serial_id.is_some(),
                Self::Reauthorization { serial_id, .. } => serial_id.is_some(),
                Self::ResetUpdate { serial_id, .. } => serial_id.is_some(),
                _ => true,
            }
    }

    /// Sets a correlated response.
    pub fn maybe_set_correlated_response(&mut self, response: &Response) {
        if self.is_correlated(response) {
            self.info_mut()
                .correlation_set_mut()
                .set_correlated_response(response.to_owned());
        }
    }

    fn set_status(&mut self, new_state: RequestStatus) {
        self.info_mut().status = new_state;
    }

    pub fn set_status_complete(&mut self) {
        self.set_status(RequestStatus::Complete);
    }

    pub fn set_status_enqueued(&mut self) {
        self.set_status(RequestStatus::Enqueued);
    }

    pub fn set_status_error(&mut self) {
        self.set_status(RequestStatus::Error);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Request:{:03}.{:03}.{}",
            self.batch_idx(),
            self.batch_item_idx(),
            self.label()
        )
    }
}

/// A data structure representing a batch of requests for system processing.
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

    pub fn next_item_idx(&self) -> usize {
        &self.requests.len() + 1
    }

    pub fn requests(&self) -> &[Request] {
        &self.requests.as_slice()
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Request> {
        &mut self.requests
    }

    pub fn enqueued_mut(&mut self) -> impl Iterator<Item = &mut Request> {
        self.requests_mut().iter_mut().filter(|r| r.is_enqueued())
    }

    pub fn complete(&mut self) -> impl Iterator<Item = &Request> {
        self.requests().iter().filter(|r| r.is_complete())
    }

    pub fn error(&mut self) -> impl Iterator<Item = &Request> {
        self.requests().iter().filter(|r| r.is_error())
    }

    pub fn new(batch_idx: usize, batch_size: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::with_capacity(batch_size * 2),
        }
    }

    pub fn is_enqueueable(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueueable())
    }

    pub fn maybe_set_response(&mut self, response: Response) {
        for request in self.enqueued_mut() {
            request.maybe_set_correlated_response(&response);
        }
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
pub enum RequestMessageBody {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}

/// Enumeration over request processing states.
#[derive(Debug, Clone)]
pub enum RequestStatus {
    // Processed successfully.
    Complete,
    // Enqueued upon system ingress queue.
    Enqueued,
    // Processed but an error occurred.
    Error,
    // Generated but awaiting processing.
    Generated,
}

impl Default for RequestStatus {
    fn default() -> Self {
        RequestStatus::Generated
    }
}

/// Enumeration over system responses dequeued from system egress queue.
#[derive(Clone, Debug)]
pub enum Response {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::ResetCheckResult),
    ResetUpdate(smpc_response::ResetUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}

impl Response {
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

#[derive(Clone, Debug)]
pub struct ResponseCorrelationSet {
    /// A correlated system response returned by node 0.
    node_0: Option<Response>,

    /// A correlated system response returned by node 1.
    node_1: Option<Response>,

    /// A correlated system response returned by node 2.
    node_2: Option<Response>,
}

impl ResponseCorrelationSet {
    /// True if all node responses have been collated.
    pub fn is_received(&self) -> bool {
        self.node_0.is_some() & self.node_1.is_some() & self.node_2.is_some()
    }

    // Sets a correlated response associated with a request.
    fn set_correlated_response(&mut self, response: Response) {
        match response.node_id() {
            0 => {
                self.node_0 = Some(response);
            }
            1 => {
                self.node_1 = Some(response);
            }
            2 => {
                self.node_2 = Some(response);
            }
            _ => panic!("Invalid node id: {}", response.node_id()),
        };
    }
}

impl Default for ResponseCorrelationSet {
    fn default() -> Self {
        Self {
            node_0: None,
            node_1: None,
            node_2: None,
        }
    }
}
