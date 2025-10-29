use std::{fmt, future::Future};

/// A batch of system requests.
#[derive(Debug)]
pub struct Batch {
    /// Ordinal batch identifier assigned by generator.
    pub batch_id: usize,
}

impl Batch {
    fn new(batch_id: usize) -> Self {
        Self { batch_id }
    }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "id={}", self.batch_id)
    }
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
