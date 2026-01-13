use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::super::typeset::{
    RequestBatch, RequestBatchKind, RequestBatchSize, ServiceClientError, UniquenessReference,
};

/// Generates batches of SMPC service requests.
pub(crate) struct RequestGenerator {
    // Count of generated batches.
    generated_batch_count: usize,

    // Parameters determining how batches are generated.
    params: RequestGeneratorParams,
}

impl RequestGenerator {
    fn batch_count(&self) -> usize {
        match &self.params {
            RequestGeneratorParams::Simple { batch_count, .. } => *batch_count,
            RequestGeneratorParams::KnownSet(batch_set) => batch_set.len(),
        }
    }

    pub(crate) fn new(params: RequestGeneratorParams) -> Self {
        Self {
            generated_batch_count: 0,
            params,
        }
    }

    /// Generates batches of request until exhausted.
    pub(crate) async fn next(&mut self) -> Result<Option<RequestBatch>, ServiceClientError> {
        if self.generated_batch_count == self.batch_count() {
            return Ok(None);
        }

        let batch = match &self.params {
            RequestGeneratorParams::Simple {
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
                            let parent =
                                push_new_uniqueness_maybe(&mut batch, kind, *known_iris_serial_id);
                            push_new(&mut batch, kind, parent);
                        }
                    }
                }
                batch
            }
            RequestGeneratorParams::KnownSet(batch_set) => {
                batch_set.get(self.next_batch_idx() - 1).unwrap().clone()
            }
        };

        self.generated_batch_count += 1;

        Ok(Some(batch))
    }

    fn next_batch_idx(&self) -> usize {
        self.generated_batch_count + 1
    }
}

/// Pushes a new request onto the batch.
fn push_new(batch: &mut RequestBatch, kind: &str, parent: Option<UniquenessReference>) {
    assert!(
        matches!(
            kind,
            smpc_request::RESET_CHECK_MESSAGE_TYPE | smpc_request::UNIQUENESS_MESSAGE_TYPE if parent.is_none()
        ) || matches!(
            kind,
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE | smpc_request::REAUTH_MESSAGE_TYPE | smpc_request::RESET_UPDATE_MESSAGE_TYPE if parent.is_some()
        ),
        "Invalid parent request association"
    );

    match kind {
        smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
            batch.push_new_identity_deletion(parent.unwrap());
        }
        smpc_request::REAUTH_MESSAGE_TYPE => {
            batch.push_new_reauthorization(parent.unwrap());
        }
        smpc_request::RESET_CHECK_MESSAGE_TYPE => {
            batch.push_new_reset_check();
        }
        smpc_request::RESET_UPDATE_MESSAGE_TYPE => {
            batch.push_new_reset_update(parent.unwrap());
        }
        smpc_request::UNIQUENESS_MESSAGE_TYPE => {
            batch.push_new_uniqueness();
        }
        _ => unreachable!(),
    }
}

// Maybe extends collection with a uniqueness request to be referenced from other requests.
fn push_new_uniqueness_maybe(
    batch: &mut RequestBatch,
    kind: &str,
    serial_id: Option<IrisSerialId>,
) -> Option<UniquenessReference> {
    match kind {
        smpc_request::RESET_CHECK_MESSAGE_TYPE | smpc_request::UNIQUENESS_MESSAGE_TYPE => None,
        smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
        | smpc_request::REAUTH_MESSAGE_TYPE
        | smpc_request::RESET_UPDATE_MESSAGE_TYPE => Some(match serial_id {
            Some(serial_id) => UniquenessReference::IrisSerialId(serial_id),
            None => UniquenessReference::SignupId(batch.push_new_uniqueness()),
        }),
        _ => panic!("Invalid request kind"),
    }
}

/// Set of variants over request generation inputs.
pub(crate) enum RequestGeneratorParams {
    /// Parameters permitting single kind batches to be generated.
    Simple {
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

#[cfg(test)]
mod tests {
    use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

    use super::{
        super::super::typeset::{RequestBatch, RequestBatchKind, RequestBatchSize},
        RequestGeneratorParams,
    };

    impl RequestGeneratorParams {
        pub fn new_1() -> Self {
            Self::Simple {
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
