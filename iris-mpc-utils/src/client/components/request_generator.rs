use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    ParentUniquenessRequest, RequestBatch, RequestBatchKind, RequestBatchSize, RequestFactory,
    ServiceClientConfig, ServiceClientError,
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

    pub fn new(config: ServiceClientConfig) -> Self {
        Self {
            generated_batch_count: 0,
            params: RequestGeneratorParams::from(config),
        }
    }

    /// Generates batches of request until exhausted.
    pub async fn next(&mut self) -> Result<Option<RequestBatch>, ServiceClientError> {
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
                let mut batch = RequestBatch::new(batch_idx, vec![]);
                for _ in 0..batch_size {
                    match batch_kind {
                        RequestBatchKind::Simple(kind) => {
                            match ParentUniquenessRequest::new_maybe(
                                &batch,
                                kind,
                                *known_iris_serial_id,
                            ) {
                                Some(parent) => {
                                    match parent {
                                        ParentUniquenessRequest::Instance(request) => {
                                            batch.push_request(RequestFactory::new_request(
                                                &batch,
                                                kind,
                                                Some(ParentUniquenessRequest::Instance(
                                                    request.clone(),
                                                )),
                                            ));
                                            batch.push_request(request);
                                        }
                                        ParentUniquenessRequest::IrisSerialId(serial_id) => {
                                            batch.push_request(RequestFactory::new_request(
                                                &batch,
                                                kind,
                                                Some(ParentUniquenessRequest::IrisSerialId(
                                                    serial_id,
                                                )),
                                            ));
                                        }
                                    };
                                }
                                None => {
                                    batch.push_request(RequestFactory::new_request(
                                        &batch, kind, None,
                                    ));
                                }
                            }
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

impl From<ServiceClientConfig> for RequestGeneratorParams {
    fn from(config: ServiceClientConfig) -> Self {
        match config {
            ServiceClientConfig::Kind {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => Self::BatchKind {
                batch_count,
                batch_size: RequestBatchSize::Static(batch_size),
                batch_kind: RequestBatchKind::from(&batch_kind),
                known_iris_serial_id,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

    use crate::client::RequestBatch;

    use super::{RequestBatchKind, RequestBatchSize, RequestGeneratorParams};

    impl RequestGeneratorParams {
        pub fn new_1() -> Self {
            Self::BatchKind {
                batch_count: 1,
                batch_size: RequestBatchSize::Static(1),
                batch_kind: RequestBatchKind::Simple(UNIQUENESS_MESSAGE_TYPE),
                known_iris_serial_id: None,
            }
        }

        fn new_2() -> Self {
            Self::KnownSet(vec![
                RequestBatch::default(),
                RequestBatch::default(),
                RequestBatch::default(),
            ])
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
