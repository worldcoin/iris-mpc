use std::{fmt, future::Future};

use iris_mpc_common::helpers::smpc_request::UniquenessRequest;

/// A batch of system requests.
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

/// The profile of a batch of system requests is related to the type of requests it contains.
#[derive(Debug, Clone)]
pub enum BatchProfile {
    /// All requests in batch are of same type.
    Simple(&'static str),
}

/// Size of each batch.
/// N.B. typcially static but dynamic sizing may be in scope for some tests.
#[derive(Debug, Clone)]
pub enum BatchSize {
    /// Fixed batch size.
    Static(usize),
}

/// Trait encapsulting iterating over a set of requests for dispatch.
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

/// Enumeration over set of supported request message types.
#[derive(Debug, Clone)]
pub enum Message {
    Uniqueness(UniquenessRequest),
}
