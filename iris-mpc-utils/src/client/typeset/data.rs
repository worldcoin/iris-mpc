use std::fmt;

use uuid;

use iris_mpc_common::{
    helpers::smpc_request::{
        self, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

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
    pub fn info(&self) -> &RequestInfo {
        match self {
            Self::IdentityDeletion { info, .. } => info,
            Self::Reauthorization { info, .. } => info,
            Self::ResetCheck { info, .. } => info,
            Self::ResetUpdate { info, .. } => info,
            Self::Uniqueness { info, .. } => info,
        }
    }

    pub fn new_identity_deletion(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::IdentityDeletion {
                info: RequestInfo::new(batch),
                serial_id: None,
                signup_id: signup_id.clone(),
            },
            _ => unreachable!(),
        }
    }

    pub fn new_reauthorisation(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::Reauthorization {
                info: RequestInfo::new(batch),
                reauth_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: signup_id.clone(),
            },
            _ => unreachable!(),
        }
    }

    pub fn new_reset_check(batch: &RequestBatch) -> Self {
        Self::ResetCheck {
            info: RequestInfo::new(batch),
            reset_check_id: uuid::Uuid::new_v4(),
        }
    }

    pub fn new_reset_update(batch: &RequestBatch, parent: &Request) -> Self {
        match parent {
            Self::Uniqueness { signup_id, .. } => Self::ResetUpdate {
                info: RequestInfo::new(batch),
                reset_update_id: uuid::Uuid::new_v4(),
                serial_id: None,
                signup_id: signup_id.clone(),
            },
            _ => unreachable!(),
        }
    }

    pub fn new_uniqueness(batch: &RequestBatch) -> Self {
        Self::Uniqueness {
            info: RequestInfo::new(batch),
            signup_id: uuid::Uuid::new_v4(),
        }
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Request:{:03}.{:03}",
            self.info().batch_idx(),
            self.info().batch_item_idx(),
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

    pub fn requests(&self) -> &Vec<Request> {
        &self.requests
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

#[derive(Clone, Debug)]
pub struct RequestInfo {
    /// Batch ordinal identifier.
    batch_idx: usize,

    /// Batch item ordinal identifier.
    batch_item_idx: usize,

    /// An identifier assigned by client for correlation purposes.
    identifier: uuid::Uuid,
}

impl RequestInfo {
    pub fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub fn batch_item_idx(&self) -> usize {
        self.batch_item_idx
    }

    pub fn identifier(&self) -> &uuid::Uuid {
        &self.identifier
    }

    pub fn new(batch: &RequestBatch) -> Self {
        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.next_item_idx(),
            identifier: uuid::Uuid::new_v4(),
        }
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Request:{:03}.{:03}.{}",
            self.batch_idx, self.batch_item_idx, self.identifier
        )
    }
}

/// Enumeration over request message body for dispatch to system egress queue.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestMessageBody {
    IdentityDeletion(smpc_request::IdentityDeletionRequest),
    Reauthorization(smpc_request::ReAuthRequest),
    ResetCheck(smpc_request::ResetCheckRequest),
    ResetUpdate(smpc_request::ResetUpdateRequest),
    Uniqueness(smpc_request::UniquenessRequest),
}
