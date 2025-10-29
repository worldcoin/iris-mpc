use super::types::{Batch, BatchIterator};

/// Encapsulates logic for generating SMPC service requests.
#[derive(Debug)]
pub struct Generator {
    // Count of generated batches.
    batch_count: usize,

    // Associated generator options.
    options: Options,
}

impl Generator {
    pub fn options(&self) -> &Options {
        &self.options
    }

    pub fn new(options: Options) -> Self {
        Self {
            batch_count: 0,
            options: options.to_owned(),
        }
    }
}

/// Options for generating SMPC service requests.
#[derive(Debug, Clone)]
pub struct Options {
    /// Number of request batches to dispatch.
    batch_count: usize,

    /// Maximum size of each batch.
    batch_size_max: usize,
}

impl Options {
    pub fn new(batch_count: usize, batch_size_max: usize) -> Self {
        Self {
            batch_count,
            batch_size_max,
        }
    }
}

impl BatchIterator for Generator {
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    async fn next_batch(&mut self) -> Option<Batch> {
        unimplemented!()
    }
}
