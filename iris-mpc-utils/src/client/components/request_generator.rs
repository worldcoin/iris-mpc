use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    ClientError, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
};

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
                let mut batch = RequestBatch::new(batch_idx, None);
                for _ in 0..batch_size {
                    match batch_kind {
                        RequestBatchKind::Simple(kind) => {
                            batch.push_child_and_maybe_parent(Request::new_and_maybe_parent(
                                &batch,
                                kind,
                                *known_iris_serial_id,
                            ));
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

impl Default for RequestGeneratorParams {
    fn default() -> Self {
        Self::BatchKind {
            batch_count: 1,
            batch_size: RequestBatchSize::Static(1),
            batch_kind: RequestBatchKind::Simple("uniqueness"),
            known_iris_serial_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

    use super::{RequestBatchKind, RequestBatchSize, RequestGeneratorParams};

    impl RequestGeneratorParams {
        fn new_1() -> Self {
            Self::BatchKind {
                batch_count: 1,
                batch_size: RequestBatchSize::Static(1),
                batch_kind: RequestBatchKind::Simple(UNIQUENESS_MESSAGE_TYPE),
                known_iris_serial_id: None,
            }
        }

        fn new_2() -> Self {
            Self::KnownSet(vec![])
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = RequestGeneratorParams::new_1();
    }

    #[tokio::test]
    async fn test_new_2() {
        let _ = RequestGeneratorParams::new_2();
    }
}
