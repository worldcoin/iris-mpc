use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    ClientError, RequestBatch, RequestBatchKind, RequestBatchSize, RequestFactory,
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

    // A known serial identifier that allows response correlation to be bypassed.
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
        for _ in 0..self.batch_size() {
            match self.batch_kind {
                RequestBatchKind::Simple(kind) => {
                    let (request, maybe_parent) =
                        RequestFactory::new_from_kind(&batch, kind, self.known_iris_serial_id);
                    batch.push_requests(request, maybe_parent);
                }
            }
        }
        self.generated_batch_count += 1;

        Ok(Some(batch))
    }
}
