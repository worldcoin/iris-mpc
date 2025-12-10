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

#[derive(Clone, Debug)]
pub enum Request {
    IdentityDeletion {
        batch_idx: usize,
        batch_item_idx: usize,
        request_id: uuid::Uuid,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
        status: RequestStatus,
    },
    Reauthorization {
        batch_idx: usize,
        batch_item_idx: usize,
        request_id: uuid::Uuid,
        reauth_id: uuid::Uuid,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
        status: RequestStatus,
    },
    ResetCheck {
        batch_idx: usize,
        batch_item_idx: usize,
        request_id: uuid::Uuid,
        reset_check_id: uuid::Uuid,
        status: RequestStatus,
    },
    ResetUpdate {
        batch_idx: usize,
        batch_item_idx: usize,
        request_id: uuid::Uuid,
        reset_update_id: uuid::Uuid,
        serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
        status: RequestStatus,
    },
    Uniqueness {
        batch_idx: usize,
        batch_item_idx: usize,
        request_id: uuid::Uuid,
        signup_id: uuid::Uuid,
        status: RequestStatus,
    },
}

impl Request {
    pub fn batch_idx(&self) -> usize {
        match self {
            Self::IdentityDeletion { batch_idx, .. }
            | Self::Reauthorization { batch_idx, .. }
            | Self::ResetCheck { batch_idx, .. }
            | Self::ResetUpdate { batch_idx, .. }
            | Self::Uniqueness { batch_idx, .. } => *batch_idx,
        }
    }

    pub fn batch_item_idx(&self) -> usize {
        match self {
            Self::IdentityDeletion { batch_item_idx, .. }
            | Self::Reauthorization { batch_item_idx, .. }
            | Self::ResetCheck { batch_item_idx, .. }
            | Self::ResetUpdate { batch_item_idx, .. }
            | Self::Uniqueness { batch_item_idx, .. } => *batch_item_idx,
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
            Self::IdentityDeletion { request_id, .. }
            | Self::Reauthorization { request_id, .. }
            | Self::ResetCheck { request_id, .. }
            | Self::ResetUpdate { request_id, .. }
            | Self::Uniqueness { request_id, .. } => request_id,
        }
    }

    pub fn status(&self) -> &RequestStatus {
        match self {
            Self::IdentityDeletion { status, .. }
            | Self::Reauthorization { status, .. }
            | Self::ResetCheck { status, .. }
            | Self::ResetUpdate { status, .. }
            | Self::Uniqueness { status, .. } => status,
        }
    }

    pub fn new_identity_deletion(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::IdentityDeletion {
                batch_idx: batch.batch_idx,
                batch_item_idx: batch.next_item_idx(),
                request_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
                status: RequestStatus::Generated,
            },
            _ => unreachable!(),
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::Reauthorization {
                batch_idx: batch.batch_idx,
                batch_item_idx: batch.next_item_idx(),
                reauth_id: uuid::Uuid::new_v4(),
                request_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
                status: RequestStatus::Generated,
            },
            _ => unreachable!(),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Self {
        Self::ResetCheck {
            batch_idx: batch.batch_idx,
            batch_item_idx: batch.next_item_idx(),
            request_id: uuid::Uuid::new_v4(),
            reset_check_id: uuid::Uuid::new_v4(),
            status: RequestStatus::Generated,
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::ResetUpdate {
                batch_idx: batch.batch_idx,
                batch_item_idx: batch.next_item_idx(),
                request_id: uuid::Uuid::new_v4(),
                reset_update_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: *signup_id,
                status: RequestStatus::Generated,
            },
            _ => unreachable!(),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Self {
        Self::Uniqueness {
            batch_idx: batch.batch_idx,
            batch_item_idx: batch.next_item_idx(),
            request_id: uuid::Uuid::new_v4(),
            signup_id: uuid::Uuid::new_v4(),
            status: RequestStatus::Generated,
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status(), RequestStatus::Complete)
    }

    pub fn is_enqueued(&self) -> bool {
        matches!(self.status(), RequestStatus::Enqueued)
    }

    pub fn is_error(&self) -> bool {
        matches!(self.status(), RequestStatus::Error)
    }

    pub fn can_enqueue(&self) -> bool {
        // True if generated and not awaiting data returned from a parent request.
        matches!(self.status(), RequestStatus::Generated)
            && match self {
                Self::IdentityDeletion { serial_id, .. } => serial_id.is_some(),
                Self::Reauthorization { serial_id, .. } => serial_id.is_some(),
                Self::ResetUpdate { serial_id, .. } => serial_id.is_some(),
                _ => true,
            }
    }

    fn set_status(&mut self, new_state: RequestStatus) {
        match self {
            Self::IdentityDeletion { status, .. }
            | Self::Reauthorization { status, .. }
            | Self::ResetCheck { status, .. }
            | Self::ResetUpdate { status, .. }
            | Self::Uniqueness { status, .. } => *status = new_state,
        }
    }

    pub fn set_status_complete(&mut self) {
        self.set_status(RequestStatus::Complete);
    }

    pub fn set_status_enqueued(&mut self) {
        self.set_status(RequestStatus::Enqueued);
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

    pub fn new(batch_idx: usize, batch_size: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::with_capacity(batch_size * 2),
        }
    }

    pub fn can_enqueue(&self) -> bool {
        self.requests.iter().any(|r| r.can_enqueue())
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
    // Has been successfully processed.
    Complete,
    // Has been enqueued upon system ingress queue.
    Enqueued,
    // Has been processed but an error occurred.
    Error,
    // Has been generated and is awaiting processing.
    Generated,
}

/// Enumeration over system response message body fetched from system egress queue.
#[derive(Debug, Clone)]
pub enum ResponseMessageBody {
    IdentityDeletion(smpc_response::IdentityDeletionResult),
    Reauthorization(smpc_response::ReAuthResult),
    ResetCheck(smpc_response::ResetCheckResult),
    ResetUpdate(smpc_response::ResetUpdateAckResult),
    Uniqueness(smpc_response::UniquenessResult),
}
