use std::{fmt, future::Future};

use async_trait::async_trait;
use uuid;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest as ReauthorisationRequest, ResetCheckRequest,
    ResetUpdateRequest, UniquenessRequest,
};

/// A service request to be dispatched to a node's ingress queue.
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
    payload: RequestPayload,

    /// Unique request identifier for correlation purposes.
    #[allow(dead_code)]
    request_id: uuid::Uuid,
}

impl Request {
    pub fn new(batch_idx: usize, batch_item_idx: usize, payload: RequestPayload) -> Self {
        Self {
            batch_idx,
            batch_item_idx,
            payload,
            request_id: uuid::Uuid::new_v4(),
        }
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

    pub fn new(batch_idx: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::new(),
        }
    }
}

impl fmt::Display for RequestBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "batch-id={}", self.batch_idx)
    }
}

/// A data structure representing the kind of request batch, i.e. the type of requests to be processed.
#[derive(Debug, Clone)]
pub enum RequestBatchKind {
    /// All requests in batch are of same type.
    Simple(&'static str),
}

/// A data structure representing inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchSize {
    /// Batch size is static.
    Static(usize),
}

/// A component responsible for dispatching request messages to system services.
#[async_trait]
pub trait RequestDispatcher {
    /// Dispatchs a batch of requests to system services.
    async fn dispatch(&self, batch: RequestBatch);
}

/// A component responsible for iterating over sets of requests.
pub trait RequestIterator {
    /// Iterator over batches of requests to be dispatched.
    ///
    /// # Returns
    ///
    /// Future that resolves to maybe a Batch.
    ///
    fn next(&mut self) -> impl Future<Output = Option<RequestBatch>> + Send;
}

/// An enumeration over a set of request payload types.
#[derive(Debug, Clone)]
pub enum RequestPayload {
    IdentityDeletion(IdentityDeletionRequest),
    Reauthorisation(ReauthorisationRequest),
    ResetCheck(ResetCheckRequest),
    ResetUpdate(ResetUpdateRequest),
    Uniqueness(UniquenessRequest),
}
