use std::fmt;

use uuid;

use iris_mpc_common::helpers::smpc_request;
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use crate::types::IrisCodeAndMaskShares;

/// Encapsualates information to dispatch a system service request.
#[derive(Debug)]
pub struct Request {
    /// Batch ordinal identifier.
    batch_idx: usize,

    /// Associated request payload.
    data: RequestData,

    /// An identifier assigned by client for correlation purposes.
    identifier: uuid::Uuid,

    /// An identifier assigned by remote service upon being enqueed.
    message_id: Option<uuid::Uuid>,

    /// Batch item ordinal identifier.
    item_idx: usize,
}

impl Request {
    pub fn data(&self) -> &RequestData {
        &self.data
    }

    pub fn identifier(&self) -> &uuid::Uuid {
        &self.identifier
    }

    pub fn message_id(&self) -> &Option<uuid::Uuid> {
        &self.message_id
    }

    pub fn new(batch_idx: usize, item_idx: usize, data: RequestData) -> Self {
        Self {
            batch_idx,
            item_idx,
            data,
            identifier: uuid::Uuid::new_v4(),
            message_id: None,
        }
    }

    pub fn set_message_id(&mut self, message_id: uuid::Uuid) {
        self.message_id = Some(message_id);
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Request:{:03}.{:03}::[{}]",
            self.batch_idx, self.item_idx, self.data
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
        let kind = match option.as_str() {
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
            }
            smpc_request::REAUTH_MESSAGE_TYPE => smpc_request::REAUTH_MESSAGE_TYPE,
            smpc_request::RESET_CHECK_MESSAGE_TYPE => smpc_request::RESET_CHECK_MESSAGE_TYPE,
            smpc_request::RESET_UPDATE_MESSAGE_TYPE => smpc_request::RESET_UPDATE_MESSAGE_TYPE,
            smpc_request::UNIQUENESS_MESSAGE_TYPE => smpc_request::UNIQUENESS_MESSAGE_TYPE,
            _ => panic!("Unsupported request batch kind"),
        };

        RequestBatchKind::Simple(kind)
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
    Reauthorization {
        shares: BothEyes<IrisCodeAndMaskShares>,
    },
    ResetCheck,
    ResetUpdate,
    Uniqueness {
        shares: BothEyes<IrisCodeAndMaskShares>,
    },
}

impl fmt::Display for RequestData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion => {
                write!(f, "IdentityDeletion")
            }
            RequestData::Reauthorization { .. } => {
                write!(f, "Reauthorisation")
            }
            RequestData::ResetCheck => {
                write!(f, "ResetCheck")
            }
            RequestData::ResetUpdate => {
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
