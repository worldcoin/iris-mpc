use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    ClientError, RequestBatch, RequestBatchKind, RequestBatchSize, RequestFactory,
};

/// Set of variants over request generation inputs.
#[derive(Debug)]
pub enum RequestGeneratorParams {
    /// Parameters permitting single kind batches to be generated.
    BatchKind {
        /// Number of request batches to generate.
        batch_count: usize,

        /// Size of each batch.
        batch_size: RequestBatchSize,

        /// Determines type of requests to be included in each batch.
        batch_kind: RequestBatchKind,

        // A known serial identifier that allows response correlation to be bypassed.
        known_iris_serial_id: Option<IrisSerialId>,
    },
    /// A pre-built known set of request batches.
    KnownSet(Vec<RequestBatch>),
}

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct RequestGenerator {
    // Count of generated batches.
    generated_batch_count: usize,

    // Parameters determining how batches are generated.
    params: RequestGeneratorParams,
}

impl RequestGenerator {
    fn batch_count(&self) -> usize {
        match &self.params {
            RequestGeneratorParams::BatchKind { batch_count, .. } => *batch_count,
            RequestGeneratorParams::KnownSet(batch_set) => batch_set.len(),
        }
    }

    pub fn new(params: RequestGeneratorParams) -> Self {
        Self {
            generated_batch_count: 0,
            params,
        }
    }

    pub fn new_from_kind(
        batch_count: usize,
        batch_kind: RequestBatchKind,
        batch_size: RequestBatchSize,
        known_iris_serial_id: Option<IrisSerialId>,
    ) -> Self {
        Self::new(RequestGeneratorParams::BatchKind {
            batch_count,
            batch_kind,
            batch_size,
            known_iris_serial_id,
        })
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ClientError> {
        if self.generated_batch_count == self.batch_count() {
            return Ok(None);
        }

        let batch = match &self.params {
            RequestGeneratorParams::BatchKind {
                batch_size,
                batch_kind,
                known_iris_serial_id,
                ..
            } => {
                let batch_idx = self.next_batch_idx();
                let batch_size = match batch_size {
                    RequestBatchSize::Static(size) => *size,
                };
                let mut batch = RequestBatch::new(batch_idx);
                for _ in 0..batch_size {
                    match batch_kind {
                        RequestBatchKind::Simple(kind) => {
                            let (request, maybe_parent) =
                                RequestFactory::new_from_kind(&batch, kind, *known_iris_serial_id);
                            batch.push_requests(request, maybe_parent);
                        }
                    }
                }
                batch
            }
            RequestGeneratorParams::KnownSet(batch_set) => {
                let batch_idx = self.next_batch_idx();
                batch_set.get(batch_idx).unwrap().clone()
            }
        };

        self.generated_batch_count += 1;

        Ok(Some(batch))
    }

    fn next_batch_idx(&self) -> usize {
        self.generated_batch_count + 1
    }
}
