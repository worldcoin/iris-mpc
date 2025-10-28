use std::fmt;

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
