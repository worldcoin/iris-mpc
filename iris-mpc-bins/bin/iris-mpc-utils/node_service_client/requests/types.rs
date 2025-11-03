use std::{fmt, future::Future};

use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest as ReauthorisationRequest, ResetCheckRequest,
    ResetUpdateRequest, UniquenessRequest,
};

/// A data structure representing a batch of requests for system processing.
#[derive(Debug)]
pub struct Batch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Requests in batch.
    requests: Vec<Payload>,
}

impl Batch {
    pub fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub fn requests(&self) -> &Vec<Payload> {
        &self.requests
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Payload> {
        &mut self.requests
    }

    pub fn new(batch_idx: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::new(),
        }
    }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "batch-id={}", self.batch_idx)
    }
}

/// A component responsible for dispatching request messages to system services.
#[async_trait]
pub trait BatchDispatcher {
    /// Dispatchs a batch of requests to system services.
    async fn dispatch_batch(&self, batch: Batch);
}

/// A component responsible for iterating over sets of requests.
pub trait BatchIterator {
    /// Count of generated batches.
    fn batch_count(&self) -> usize;

    /// Iterator over batches of requests to be dispatched.
    ///
    /// # Returns
    ///
    /// Future that resolves to maybe a Batch.
    ///
    fn next_batch(&mut self) -> impl Future<Output = Option<Batch>> + Send;
}

/// A data structure representing the kind of request batch, i.e. the type of requests to be processed.
#[derive(Debug, Clone)]
pub enum BatchKind {
    /// All requests in batch are of same type.
    Simple(&'static str),
}

/// A data structure representing inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum BatchSize {
    /// Batch size is static.
    Static(usize),
}

/// An enumeration over a set of request payload types.
#[derive(Debug, Clone)]
pub enum Payload {
    IdentityDeletion(IdentityDeletionRequest),
    Reauthorisation(ReauthorisationRequest),
    ResetCheck(ResetCheckRequest),
    ResetUpdate(ResetUpdateRequest),
    Uniqueness(UniquenessRequest),
}

/// A component responsible for generating request payload instances.
pub trait PayloadFactory {
    fn create_payload(&self, batch_idx: usize, item_idx: usize) -> Payload;
}
