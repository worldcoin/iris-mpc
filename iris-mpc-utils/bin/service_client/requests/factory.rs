use std::panic;

use iris_mpc_common::helpers::smpc_request::{UniquenessRequest, UNIQUENESS_MESSAGE_TYPE};

use super::types::{BatchProfile, Message};

#[derive(Debug)]
pub struct Factory {
    // Associated options.
    options: FactoryOptions,
}

impl Factory {
    pub fn new(options: FactoryOptions) -> Self {
        Self { options }
    }

    pub fn create_request_message(&self, batch_idx: usize, item_idx: usize) -> Message {
        match self.options.batch_profile {
            BatchProfile::Simple(kind) => match kind {
                UNIQUENESS_MESSAGE_TYPE => {
                    Message::Uniqueness(create_request_uniqueness(batch_idx, item_idx))
                }
                _ => panic!("Unsupported batch profile"),
            },
        }
    }
}

/// Options for creating SMPC service requests.
#[derive(Debug, Clone)]
pub struct FactoryOptions {
    /// Type of requests to be included in each batch.
    batch_profile: BatchProfile,
}

impl FactoryOptions {
    pub fn new(batch_profile: BatchProfile) -> Self {
        Self { batch_profile }
    }
}

/// TODO: move to a dedicated factory component ?
pub fn create_request_uniqueness(_batch_idx: usize, _item_idx: usize) -> UniquenessRequest {
    unimplemented!()
}
