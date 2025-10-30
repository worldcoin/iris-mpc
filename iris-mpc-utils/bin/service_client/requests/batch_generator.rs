use super::{
    factory::Factory,
    types::{Batch, BatchIterator, BatchSize},
};

/// Encapsulates logic for generating batches of SMPC service requests.
#[derive(Debug)]
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // Associated component that instantiates request instances.
    factory: Factory,

    // Associated options.
    options: BatchGeneratorOptions,
}

impl BatchGenerator {
    pub fn new(options: BatchGeneratorOptions, factory: Factory) -> Self {
        Self {
            factory,
            batch_count: 0,
            options: options.to_owned(),
        }
    }
}

/// Options for request batch generation.
#[derive(Debug, Clone)]
pub struct BatchGeneratorOptions {
    /// Size of each batch.
    batch_size: BatchSize,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl BatchGeneratorOptions {
    pub fn new(batch_size: BatchSize, n_batches: usize) -> Self {
        Self {
            batch_size,
            n_batches,
        }
    }
}

impl BatchIterator for BatchGenerator {
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    async fn next_batch(&mut self) -> Option<Batch> {
        if self.batch_count() == self.options.n_batches {
            return None;
        }

        let batch_idx = self.batch_count() + 1;
        let mut batch = Batch::new(batch_idx);

        match self.options.batch_size {
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
