use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    ClientError, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator {
    /// Number of request batches to generate.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: RequestBatchKind,

    /// Size of each batch.
    batch_size: RequestBatchSize,

    // Count of generated batches.
    generated_batch_count: usize,

    /// A known Iris serial identifier used to by full response correlation.
    /// Note: this is a temporary field until correlation is fully supported.
    known_iris_serial_id: Option<IrisSerialId>,
}

impl RequestGenerator {
    fn batch_size(&self) -> usize {
        match self.batch_size {
            RequestBatchSize::Static(size) => size,
        }
    }

    pub fn new(
        batch_count: usize,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        known_iris_serial_id: Option<IrisSerialId>,
    ) -> Self {
        Self {
            generated_batch_count: 0,
            batch_count,
            batch_kind,
            batch_size,
            known_iris_serial_id,
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ClientError> {
        if self.generated_batch_count == self.batch_count {
            return Ok(None);
        }

        let batch_idx = self.generated_batch_count + 1;
        let mut batch = RequestBatch::new(batch_idx, self.batch_size());
        for batch_item_idx in 1..(self.batch_size() + 1) {
            batch.requests_mut().push(match self.batch_kind {
                RequestBatchKind::Simple(batch_kind) => Request::new(
                    batch_idx,
                    batch_item_idx,
                    batch_kind,
                    self.known_iris_serial_id,
                ),
            });
        }
        self.generated_batch_count += 1;

        tracing::info!("{} :: Instantiated", batch);

        Ok(Some(batch))
    }
}
