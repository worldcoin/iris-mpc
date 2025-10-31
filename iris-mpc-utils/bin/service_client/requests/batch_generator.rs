use super::{
    factory::Factory,
    types::{Batch, BatchIterator, BatchSize},
};

/// Encapsulates logic for generating batches of SMPC service requests.
#[derive(Debug)]
pub struct Generator {
    // Count of generated batches.
    batch_count: usize,

    /// Size of each batch.
    batch_size: BatchSize,

    // Associated component that instantiates request instances.
    factory: Factory,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl Generator {
    pub fn new(batch_size: BatchSize, factory: Factory, n_batches: usize) -> Self {
        Self {
            batch_count: 0,
            batch_size,
            factory,
            n_batches,
        }
    }
}

impl BatchIterator for Generator {
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    async fn next_batch(&mut self) -> Option<Batch> {
        if self.batch_count() == self.n_batches {
            return None;
        }

        let batch_idx = self.batch_count() + 1;
        let mut batch = Batch::new(batch_idx);

        match self.batch_size {
            BatchSize::Static(size) => {
                for item_idx in 1..(size + 1) {
                    let item = self.factory.create_request_message(batch_idx, item_idx);
                    batch.requests_mut().push(item);
                }
            }
        }

        self.batch_count += 1;

        Some(batch)
    }
}
