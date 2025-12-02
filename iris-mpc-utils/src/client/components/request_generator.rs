use super::super::typeset::{
    ClientError, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator {
    // Count of generated batches.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: RequestBatchKind,

    /// Size of each batch.
    batch_size: RequestBatchSize,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl RequestGenerator {
    fn batch_size(&self) -> usize {
        match self.batch_size {
            RequestBatchSize::Static(size) => size,
        }
    }

    pub fn new(
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        n_batches: usize,
    ) -> Self {
        Self {
            batch_count: 0,
            batch_kind,
            batch_size,
            n_batches,
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ClientError> {
        if self.batch_count == self.n_batches {
            return Ok(None);
        }

        let batch_idx = self.batch_count + 1;
        let mut batch = RequestBatch::new(batch_idx, self.batch_size());
        tracing::info!("{} :: Instantiated", batch);

        for batch_item_idx in 1..(self.batch_size() + 1) {
            batch.requests_mut().push(match self.batch_kind {
                RequestBatchKind::Simple(batch_kind) => {
                    Request::new(batch_idx, batch_item_idx, batch_kind)
                }
            });
        }

        self.batch_count += 1;

        Ok(Some(batch))
    }
}
