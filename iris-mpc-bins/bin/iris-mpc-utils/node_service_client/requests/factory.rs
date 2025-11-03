use std::panic;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::types::{BatchKind, Payload, PayloadFactory};

/// A service request factory.
#[derive(Debug)]
pub struct Factory {
    /// Determines type of requests to be included in each batch.
    batch_profile: BatchKind,
}

impl PayloadFactory for Factory {
    fn create_payload(&self, batch_idx: usize, item_idx: usize) -> Payload {
        match self.batch_profile {
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
}

impl Factory {
    pub fn new(batch_profile: BatchKind) -> Self {
        Self { batch_profile }
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
