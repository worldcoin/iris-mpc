use std::panic;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::types::{BatchProfile, Message, MessageFactory};

impl MessageFactory for Factory {
    fn create_message(&self, batch_idx: usize, item_idx: usize) -> Message {
        match self.batch_profile {
            BatchProfile::Simple(kind) => FactoryByKind::create_message(batch_idx, item_idx, kind),
        }
    }
}

pub struct Factory {
    /// Determines type of requests to be included in each batch.
    batch_profile: BatchProfile,
}

impl Factory {
    pub fn new(batch_profile: BatchProfile) -> Self {
        Self { batch_profile }
    }
}

/// A component responsible for instantiating SMPC requests by kind.
struct FactoryByKind {}

impl FactoryByKind {
    fn create_message(batch_idx: usize, item_idx: usize, request_kind: &str) -> Message {
        match request_kind {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                Message::IdentityDeletion(Self::create_identity_deletion(batch_idx, item_idx))
            }
            REAUTH_MESSAGE_TYPE => {
                Message::Reauthorisation(Self::create_reauthorisation(batch_idx, item_idx))
            }
            RESET_CHECK_MESSAGE_TYPE => {
                Message::ResetCheck(Self::create_reset_check(batch_idx, item_idx))
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                Message::ResetUpdate(Self::create_reset_update(batch_idx, item_idx))
            }
            UNIQUENESS_MESSAGE_TYPE => {
                Message::Uniqueness(Self::create_uniqueness(batch_idx, item_idx))
            }
            _ => panic!("Unsupported request kind: {}", request_kind),
        }
    }

    fn create_identity_deletion(_batch_idx: usize, _item_idx: usize) -> IdentityDeletionRequest {
        unimplemented!("create_identity_deletion")
    }

    fn create_reauthorisation(_batch_idx: usize, _item_idx: usize) -> ReAuthRequest {
        unimplemented!("create_reauthorisation")
    }

    fn create_reset_check(_batch_idx: usize, _item_idx: usize) -> ResetCheckRequest {
        unimplemented!("create_reset_check")
    }

    fn create_reset_update(_batch_idx: usize, _item_idx: usize) -> ResetUpdateRequest {
        unimplemented!("create_reset_update")
    }

    fn create_uniqueness(_batch_idx: usize, _item_idx: usize) -> UniquenessRequest {
        unimplemented!("create_uniqueness")
    }
}
