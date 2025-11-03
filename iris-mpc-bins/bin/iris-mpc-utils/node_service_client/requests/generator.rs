use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::types::{Batch, BatchKind, BatchSize, Payload, RequestIterator};

/// Encapsulates logic for generating batches of SMPC service request messages.
#[derive(Debug)]
pub struct Generator {
    // Count of generated batches.
    batch_count: usize,

    /// Determines type of requests to be included in each batch.
    batch_kind: BatchKind,

    /// Size of each batch.
    batch_size: BatchSize,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl Generator {
    pub fn new(batch_kind: BatchKind, batch_size: BatchSize, n_batches: usize) -> Self {
        Self {
            batch_count: 0,
            batch_kind,
            batch_size,
            n_batches,
        }
    }

    fn create_payload(&self, batch_idx: usize, item_idx: usize) -> Payload {
        match self.batch_kind {
            BatchKind::Simple(kind) => match kind {
                IDENTITY_DELETION_MESSAGE_TYPE => {
                    Payload::IdentityDeletion(self.create_identity_deletion(batch_idx, item_idx))
                }
                REAUTH_MESSAGE_TYPE => {
                    Payload::Reauthorisation(self.create_reauthorisation(batch_idx, item_idx))
                }
                RESET_CHECK_MESSAGE_TYPE => {
                    Payload::ResetCheck(self.create_reset_check(batch_idx, item_idx))
                }
                RESET_UPDATE_MESSAGE_TYPE => {
                    Payload::ResetUpdate(self.create_reset_update(batch_idx, item_idx))
                }
                UNIQUENESS_MESSAGE_TYPE => {
                    Payload::Uniqueness(self.create_uniqueness(batch_idx, item_idx))
                }
                _ => panic!("Unsupported request kind: {}", kind),
            },
        }
    }

    fn create_identity_deletion(
        &self,
        _batch_idx: usize,
        _item_idx: usize,
    ) -> IdentityDeletionRequest {
        unimplemented!("create_identity_deletion")
    }

    fn create_reauthorisation(&self, _batch_idx: usize, _item_idx: usize) -> ReAuthRequest {
        unimplemented!("create_reauthorisation")
    }

    fn create_reset_check(&self, _batch_idx: usize, _item_idx: usize) -> ResetCheckRequest {
        unimplemented!("create_reset_check")
    }

    fn create_reset_update(&self, _batch_idx: usize, _item_idx: usize) -> ResetUpdateRequest {
        unimplemented!("create_reset_update")
    }

    fn create_uniqueness(&self, _batch_idx: usize, _item_idx: usize) -> UniquenessRequest {
        UniquenessRequest {
            batch_size: None,
            disable_anonymized_stats: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            or_rule_serial_ids: None,
            s3_key: "test_s3_key".to_string(),
            signup_id: "test_signup_id".to_string(),
            skip_persistence: None,
        }
    }
}

impl RequestIterator for Generator {
    async fn next(&mut self) -> Option<Batch> {
        if self.batch_count == self.n_batches {
            return None;
        }

        let batch_idx = self.batch_count + 1;
        let mut batch = Batch::new(batch_idx);

        match self.batch_size {
            BatchSize::Static(size) => {
                for item_idx in 1..(size + 1) {
                    batch
                        .requests_mut()
                        .push(self.create_payload(batch_idx, item_idx));
                }
            }
        }

        self.batch_count += 1;

        Some(batch)
    }
}
