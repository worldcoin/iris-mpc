use super::types::{Batch, BatchIterator, BatchSize, MessageFactory};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct Generator<F>
where
    F: MessageFactory,
{
    // Count of generated batches.
    batch_count: usize,

    /// Size of each batch.
    batch_size: BatchSize,

    // Associated component that instantiates messages instances.
    message_factory: F,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl<F> Generator<F>
where
    F: MessageFactory,
{
    pub fn new(batch_size: BatchSize, message_factory: F, n_batches: usize) -> Self {
        Self {
            batch_count: 0,
            batch_size,
            message_factory,
            n_batches,
        }
    }
}

impl<F> BatchIterator for Generator<F>
where
    F: MessageFactory + Send,
{
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
                    let item = self.message_factory.create_message(batch_idx, item_idx);
                    batch.requests_mut().push(item);
                }
            }
        }

        self.batch_count += 1;

        Some(batch)
    }
}
