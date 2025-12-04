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
        known_iris_serial_id: Option<IrisSerialId>,
        signup_id: uuid::Uuid,
    },
    Reauthorization {
        info: RequestInfo,
        known_iris_serial_id: Option<IrisSerialId>,
        reauth_id: uuid::Uuid,
        signup_id: uuid::Uuid,
    },
    ResetCheck {
        info: RequestInfo,
        known_iris_serial_id: Option<IrisSerialId>,
        reset_id: uuid::Uuid,
        signup_id: uuid::Uuid,
    },
    ResetUpdate {
        info: RequestInfo,
        known_iris_serial_id: Option<IrisSerialId>,
        reset_id: uuid::Uuid,
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

    pub fn new(
        batch_idx: usize,
        batch_item_idx: usize,
        batch_kind: &'static str,
        known_iris_serial_id: Option<IrisSerialId>,
    ) -> Self {
        match batch_kind {
            IDENTITY_DELETION_MESSAGE_TYPE => Self::IdentityDeletion {
                info: RequestInfo::new(batch_idx, batch_item_idx),
                known_iris_serial_id,
                signup_id: uuid::Uuid::new_v4(),
            },
            REAUTH_MESSAGE_TYPE => Self::Reauthorization {
                info: RequestInfo::new(batch_idx, batch_item_idx),
                known_iris_serial_id,
                reauth_id: uuid::Uuid::new_v4(),
                signup_id: uuid::Uuid::new_v4(),
            },
            RESET_CHECK_MESSAGE_TYPE => Self::ResetCheck {
                info: RequestInfo::new(batch_idx, batch_item_idx),
                known_iris_serial_id,
                reset_id: uuid::Uuid::new_v4(),
                signup_id: uuid::Uuid::new_v4(),
            },
            RESET_UPDATE_MESSAGE_TYPE => Self::ResetUpdate {
                info: RequestInfo::new(batch_idx, batch_item_idx),
                known_iris_serial_id,
                reset_id: uuid::Uuid::new_v4(),
                signup_id: uuid::Uuid::new_v4(),
            },
            UNIQUENESS_MESSAGE_TYPE => Self::Uniqueness {
                info: RequestInfo::new(batch_idx, batch_item_idx),
                signup_id: uuid::Uuid::new_v4(),
            },
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion { .. } => {
                write!(
                    f,
                    "Request:{:03}.{:03}.IdentityDeletion",
                    self.info().batch_idx(),
                    self.info().batch_item_idx(),
                )
            }
            Self::Reauthorization { .. } => {
                write!(
                    f,
                    "Request:{:03}.{:03}.Reauthorization",
                    self.info().batch_idx(),
                    self.info().batch_item_idx(),
                )
            }
            Self::ResetCheck { .. } => {
                write!(
                    f,
                    "Request:{:03}.{:03}.ResetCheck",
                    self.info().batch_idx(),
                    self.info().batch_item_idx(),
                )
            }
            Self::ResetUpdate { .. } => {
                write!(
                    f,
                    "Request:{:03}.{:03}.ResetUpdate",
                    self.info().batch_idx(),
                    self.info().batch_item_idx(),
                )
            }
            Self::Uniqueness { .. } => {
                write!(
                    f,
                    "Request:{:03}.{:03}.Uniqueness",
                    self.info().batch_idx(),
                    self.info().batch_item_idx(),
                )
            }
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

    pub fn new(batch_idx: usize, batch_item_idx: usize) -> Self {
        Self {
            batch_idx,
            batch_item_idx,
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

    pub fn requests(&self) -> &Vec<Request> {
        &self.requests
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Request> {
        &mut self.requests
    }

    pub fn new(batch_idx: usize, batch_size: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::with_capacity(batch_size),
        }
    }
}

impl fmt::Display for RequestBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request-Batch:{:03}", self.batch_idx)
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

/// Enumeration over generated request data.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestData {
    IdentityDeletion,
    Reauthorization,
    ResetCheck,
    ResetUpdate,
    Uniqueness,
}

impl fmt::Display for RequestData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion { .. } => {
                write!(f, "IdentityDeletion")
            }
            RequestData::Reauthorization { .. } => {
                write!(f, "Reauthorisation")
            }
            RequestData::ResetCheck { .. } => {
                write!(f, "ResetCheck")
            }
            RequestData::ResetUpdate { .. } => {
                write!(f, "ResetUpdate")
            }
            RequestData::Uniqueness { .. } => {
                write!(f, "Uniqueness")
            }
        }
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
