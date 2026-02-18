use iris_mpc_common::IrisSerialId;

use super::super::typeset::{BatchKind, PendingItem, RequestBatch, RequestBatchSet, ServiceClientError};
use crate::client::options::{
    Parent, RequestBatchOptions, RequestOptions, RequestPayloadOptions, ServiceClientOptions,
};

/// Generates batches of SMPC service requests.
pub(crate) struct RequestGenerator {
    // Count of generated batches.
    generated_batch_count: usize,

    // Parameters determining how batches are generated.
    config: RequestGeneratorConfig,
}

impl RequestGenerator {
    fn batch_count(&self) -> usize {
        match &self.config {
            RequestGeneratorConfig::Simple { batch_count, .. } => *batch_count,
            RequestGeneratorConfig::Complex(batch_set) => batch_set.batches().len(),
        }
    }

    pub(crate) fn new(config: RequestGeneratorConfig) -> Self {
        Self {
            generated_batch_count: 0,
            config,
        }
    }

    pub(crate) fn from_options(opts: &ServiceClientOptions) -> Result<Self, ServiceClientError> {
        let config = RequestGeneratorConfig::try_from_options(opts)?;
        Ok(Self::new(config))
    }

    /// Generates batches of request until exhausted.
    pub(crate) async fn next(&mut self) -> Result<Option<RequestBatch>, ServiceClientError> {
        if self.generated_batch_count == self.batch_count() {
            return Ok(None);
        }

        let batch = match &self.config {
            RequestGeneratorConfig::Complex(batch_set) => batch_set
                .batches()
                .get(self.next_batch_idx() - 1)
                .unwrap()
                .clone(),
            RequestGeneratorConfig::Simple {
                batch_size,
                batch_kind,
                known_iris_serial_id,
                ..
            } => {
                let batch_idx = self.next_batch_idx();
                let mut batch = RequestBatch::new(batch_idx, vec![]);
                for _ in 0..*batch_size {
                    let parent_result =
                        push_new_uniqueness_maybe(&mut batch, batch_kind, *known_iris_serial_id);
                    push_new(&mut batch, batch_kind, parent_result);
                }
                batch
            }
        };

        self.generated_batch_count += 1;

        Ok(Some(batch))
    }

    fn next_batch_idx(&self) -> usize {
        self.generated_batch_count + 1
    }
}

/// Result of maybe-creating a uniqueness parent request.
enum ParentResult {
    /// Parent serial ID is already known.
    Known(IrisSerialId),
    /// Parent is an intra-batch Uniqueness; child should wait via PendingItem.
    Pending(String),
}

/// Maybe extends the batch with a Uniqueness request for child requests to reference.
fn push_new_uniqueness_maybe(
    batch: &mut RequestBatch,
    kind: &BatchKind,
    serial_id: Option<IrisSerialId>,
) -> Option<ParentResult> {
    if !kind.requires_parent() {
        return None;
    }

    Some(match serial_id {
        Some(serial_id) => ParentResult::Known(serial_id),
        None => {
            let signup_id = batch.push_new_uniqueness(None, None);
            ParentResult::Pending(signup_id.to_string())
        }
    })
}

/// Pushes a new request (or pending item) onto the batch.
fn push_new(batch: &mut RequestBatch, kind: &BatchKind, parent: Option<ParentResult>) {
    assert_eq!(
        kind.requires_parent(),
        parent.is_some(),
        "Invalid parent request association for {:?}",
        kind
    );

    match (kind, parent) {
        (BatchKind::Uniqueness, None) => {
            batch.push_new_uniqueness(None, None);
        }
        (BatchKind::ResetCheck, None) => {
            batch.push_new_reset_check(None, None);
        }
        (BatchKind::IdentityDeletion, Some(ParentResult::Known(serial_id))) => {
            batch.push_new_identity_deletion(serial_id, None);
        }
        (BatchKind::Reauth, Some(ParentResult::Known(serial_id))) => {
            batch.push_new_reauthorization(serial_id, None, None);
        }
        (BatchKind::ResetUpdate, Some(ParentResult::Known(serial_id))) => {
            batch.push_new_reset_update(serial_id, None, None);
        }
        (BatchKind::IdentityDeletion, Some(ParentResult::Pending(parent_key))) => {
            batch.push_pending(PendingItem::new(
                parent_key.clone(),
                RequestOptions::new(
                    None,
                    RequestPayloadOptions::IdentityDeletion {
                        parent: Parent::Label(parent_key),
                    },
                ),
            ));
        }
        (BatchKind::Reauth, Some(ParentResult::Pending(parent_key))) => {
            batch.push_pending(PendingItem::new(
                parent_key.clone(),
                RequestOptions::new(
                    None,
                    RequestPayloadOptions::Reauthorisation {
                        iris_pair: None,
                        parent: Parent::Label(parent_key),
                    },
                ),
            ));
        }
        (BatchKind::ResetUpdate, Some(ParentResult::Pending(parent_key))) => {
            batch.push_pending(PendingItem::new(
                parent_key.clone(),
                RequestOptions::new(
                    None,
                    RequestPayloadOptions::ResetUpdate {
                        iris_pair: None,
                        parent: Parent::Label(parent_key),
                    },
                ),
            ));
        }
        _ => unreachable!("Invalid combination of BatchKind and parent"),
    }
}

/// Set of variants over request generation inputs.
pub(crate) enum RequestGeneratorConfig {
    /// A pre-built known set of request batches.
    Complex(RequestBatchSet),
    /// Parameters permitting single kind batches to be generated.
    Simple {
        /// Number of request batches to generate.
        batch_count: usize,

        /// Size of each batch.
        batch_size: usize,

        /// Determines type of requests to be included in each batch.
        batch_kind: BatchKind,

        // A known serial identifier that allows response correlation to be bypassed.
        known_iris_serial_id: Option<IrisSerialId>,
    },
}

impl RequestGeneratorConfig {
    pub(crate) fn try_from_options(
        opts: &ServiceClientOptions,
    ) -> Result<Self, ServiceClientError> {
        match opts.request_batch() {
            RequestBatchOptions::Complex {
                batches: opts_batches,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Complex");
                Ok(Self::Complex(RequestBatchSet::from_options(opts_batches)))
            }
            RequestBatchOptions::Simple {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Simple");
                Ok(Self::Simple {
                    batch_count: *batch_count,
                    batch_size: *batch_size,
                    batch_kind: BatchKind::from_str(batch_kind).ok_or_else(|| {
                        ServiceClientError::InvalidOptions(format!(
                            "Unsupported batch kind: {}",
                            batch_kind
                        ))
                    })?,
                    known_iris_serial_id: *known_iris_serial_id,
                })
            }
        }
    }
}
