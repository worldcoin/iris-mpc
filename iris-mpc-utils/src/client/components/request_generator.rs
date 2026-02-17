use iris_mpc_common::IrisSerialId;

use super::super::typeset::{
    BatchKind, RequestBatch, RequestBatchSet, ServiceClientError, UniquenessRequestDescriptor,
};
use crate::client::options::{RequestBatchOptions, ServiceClientOptions};

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
        let mut config = RequestGeneratorConfig::try_from_options(opts)?;
        config.set_child_parent_descriptors_from_labels();
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
                    let parent =
                        push_new_uniqueness_maybe(&mut batch, batch_kind, *known_iris_serial_id);
                    push_new(&mut batch, batch_kind, parent);
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

/// Pushes a new request onto the batch.
fn push_new(
    batch: &mut RequestBatch,
    kind: &BatchKind,
    parent: Option<UniquenessRequestDescriptor>,
) {
    assert_eq!(
        kind.requires_parent(),
        parent.is_some(),
        "Invalid parent request association for {:?}",
        kind
    );

    match kind {
        BatchKind::IdentityDeletion => {
            batch.push_new_identity_deletion(parent.unwrap(), None, None);
        }
        BatchKind::Reauth => {
            batch.push_new_reauthorization(parent.unwrap(), None, None, None);
        }
        BatchKind::ResetCheck => {
            batch.push_new_reset_check(None, None);
        }
        BatchKind::ResetUpdate => {
            batch.push_new_reset_update(parent.unwrap(), None, None, None);
        }
        BatchKind::Uniqueness => {
            batch.push_new_uniqueness(None, None);
        }
    }
}

/// Maybe extends collection with a uniqueness request to be referenced from other requests.
fn push_new_uniqueness_maybe(
    batch: &mut RequestBatch,
    kind: &BatchKind,
    serial_id: Option<IrisSerialId>,
) -> Option<UniquenessRequestDescriptor> {
    if !kind.requires_parent() {
        return None;
    }

    Some(match serial_id {
        Some(serial_id) => UniquenessRequestDescriptor::IrisSerialId(serial_id),
        None => UniquenessRequestDescriptor::SignupId(batch.push_new_uniqueness(None, None)),
    })
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

    // Reassigns parent descriptors.
    pub(crate) fn set_child_parent_descriptors_from_labels(&mut self) {
        if let RequestGeneratorConfig::Complex(batch_set) = self {
            batch_set.set_child_parent_descriptors_from_labels();
        }
    }
}
