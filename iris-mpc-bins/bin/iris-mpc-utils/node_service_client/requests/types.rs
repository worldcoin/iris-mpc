use std::{fmt, future::Future};

use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest,
};

/// A data structure representing a batch of requests for system processing.
#[derive(Debug)]
pub struct Batch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Requests in batch.
    requests: Vec<Message>,
}

impl Batch {
    pub fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub fn requests(&self) -> &Vec<Message> {
        &self.requests
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Message> {
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

/// A data structure representing the profile of a request batch, i.e. the type of requests to be processed.
#[derive(Debug, Clone)]
pub enum BatchProfile {
    /// All requests in batch are of same type.
    Simple(&'static str),
}

/// A data structure representing inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum BatchSize {
    /// Batch size is static.
    Static(usize),
}

/// An enumeration over a set of request message types.
#[derive(Debug, Clone)]
pub enum Message {
    IdentityDeletion(IdentityDeletionRequest),
    Reauthorisation(ReAuthRequest),
    ResetCheck(ResetCheckRequest),
    ResetUpdate(ResetUpdateRequest),
    Uniqueness(UniquenessRequest),
}

/// A component responsible for generating request message instances.
pub trait MessageFactory {
    fn create_message(&self, batch_idx: usize, item_idx: usize) -> Message;
}
