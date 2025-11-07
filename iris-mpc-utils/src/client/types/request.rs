use std::fmt;

use uuid;

use iris_mpc_cpu::execution::hawk_main::BothEyes;

use crate::types::IrisCodeAndMaskShares;

/// Encapsualates information to dispatch a system service request.
#[derive(Debug)]
pub struct Request {
    /// Batch ordinal identifier.
    #[allow(dead_code)]
    batch_idx: usize,

    /// Batch item ordinal identifier.
    #[allow(dead_code)]
    batch_item_idx: usize,

    /// Associated request payload.
    #[allow(dead_code)]
    data: RequestData,

    /// Unique request identifier for correlation purposes.
    #[allow(dead_code)]
    identifier: uuid::Uuid,
}

impl Request {
    pub fn batch_idx(&self) -> &usize {
        &self.batch_idx
    }

    pub fn batch_item_idx(&self) -> &usize {
        &self.batch_item_idx
    }

    pub fn data(&self) -> &RequestData {
        &self.data
    }

    pub fn identifier(&self) -> &uuid::Uuid {
        &self.identifier
    }

    pub fn new(batch_idx: usize, batch_item_idx: usize, data: RequestData) -> Self {
        Self {
            batch_idx,
            batch_item_idx,
            data,
            identifier: uuid::Uuid::new_v4(),
        }
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "request: batch-id={} item-id={} type={}",
            self.batch_idx, self.batch_item_idx, self.data
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
        write!(f, "batch-id={}", self.batch_idx)
    }
}

/// Encapsulates inputs used to derive the kind of request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchKind {
    /// All requests are of same type.
    Simple(&'static str),
}

/// Encapsulates inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchSize {
    /// Batch size is static.
    Static(usize),
}

/// Enumeration over data associated with a request.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum RequestData {
    IdentityDeletion,
    Reauthorisation,
    ResetCheck,
    ResetUpdate,
    Uniqueness(RequestDataUniqueness),
}

#[derive(Debug, Clone)]
pub struct RequestDataUniqueness {
    shares: BothEyes<IrisCodeAndMaskShares>,
}

impl RequestDataUniqueness {
    pub fn iris_code_and_mask_shares_both_eyes(&self) -> &BothEyes<IrisCodeAndMaskShares> {
        &self.shares
    }

    pub fn new(shares: BothEyes<IrisCodeAndMaskShares>) -> Self {
        Self { shares }
    }
}

impl fmt::Display for RequestData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IdentityDeletion => {
                write!(f, "AMPC.IdentityDeletion")
            }
            RequestData::Reauthorisation => {
                write!(f, "AMPC.Reauthorisation")
            }
            RequestData::ResetCheck => {
                write!(f, "AMPC.ResetCheck")
            }
            RequestData::ResetUpdate => {
                write!(f, "AMPC.ResetUpdate")
            }
            RequestData::Uniqueness(_) => {
                write!(f, "AMPC.Uniqueness")
            }
        }
    }
}
