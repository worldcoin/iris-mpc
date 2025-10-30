use std::panic;

use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

use super::types::{Batch, BatchIterator, BatchProfile, BatchSize, Message};

/// Encapsulates logic for generating SMPC service requests.
#[derive(Debug)]
pub struct Generator {
    // Count of generated batches.
    batch_count: usize,

    // Associated generator options.
    options: Options,
}

impl Generator {
    pub fn new(options: Options) -> Self {
        Self {
            batch_count: 0,
            options: options.to_owned(),
        }
    }
}

/// Options for generating SMPC service requests.
#[derive(Debug, Clone)]
pub struct Options {
    /// Type of requests to be included in each batch.
    batch_profile: BatchProfile,

    /// Size of each batch.
    batch_size: BatchSize,

    /// Number of request batches to generate.
    n_batches: usize,
}

impl Options {
    pub fn new(batch_profile: BatchProfile, batch_size: BatchSize, n_batches: usize) -> Self {
        Self {
            batch_profile,
            batch_size,
            n_batches,
        }
    }
}

impl BatchIterator for Generator {
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    async fn next_batch(&mut self) -> Option<Batch> {
        if self.batch_count() == self.options.n_batches {
            return None;
        }

        self.batch_count += 1;
        Some(Batch::new(self.batch_count()))
    }
}

/// Creates a batch of requests.
/// TODO: return a stream rather than a vector
/// TODO: create a dedicated factory component
fn create_batch(batch_id: usize, batch_profile: &BatchProfile, batch_size: &BatchSize) -> Batch {
    let mut batch = Batch::new(batch_id);
    match batch_size {
        BatchSize::Static(size) => {
            for item_id in 0..*size {
                batch
                    .requests_mut()
                    .push(create_batch_item(batch_id, item_id + 1, batch_profile))
            }
        }
    }

    batch
}

fn create_batch_item(_batch_id: usize, _item_id: usize, _batch_profile: &BatchProfile) -> Message {
    // match batch_profile {
    //     BatchProfile::Simple(kind) => match kind {
    //         UNIQUENESS_MESSAGE_TYPE => result.push(Batch::new(1)),
    //         _ => panic!("Unsupported batch profile"),
    //     },
    // }
    unimplemented!()
}
