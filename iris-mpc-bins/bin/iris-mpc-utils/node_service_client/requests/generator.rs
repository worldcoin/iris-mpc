use super::types::{Batch, BatchSize, PayloadFactory, RequestIterator};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct Generator<F>
where
    F: PayloadFactory,
{
    // Count of generated batches.
    batch_count: usize,

    /// Size of each batch.
    batch_size: BatchSize,

    // Associated component that instantiates messages instances.
    request_payload_factory: F,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl<F> Generator<F>
where
    F: PayloadFactory,
{
    pub fn new(batch_size: BatchSize, n_batches: usize, request_payload_factory: F) -> Self {
        Self {
            batch_count: 0,
            batch_size,
            n_batches,
            request_payload_factory,
        }
    }
}

impl<F> RequestIterator for Generator<F>
where
    F: PayloadFactory + Send,
{
    async fn next(&mut self) -> Option<Batch> {
        if self.batch_count == self.n_batches {
            return None;
        }

        let batch_idx = self.batch_count + 1;
        let mut batch = Batch::new(batch_idx);

        match self.batch_size {
            BatchSize::Static(size) => {
                for item_idx in 1..(size + 1) {
                    let item = self
                        .request_payload_factory
                        .create_payload(batch_idx, item_idx);
                    batch.requests_mut().push(item);
                }
            }
        }

        self.batch_count += 1;

        Some(batch)
    }
}
