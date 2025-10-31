use async_trait::async_trait;

use super::types::{Batch, BatchDispatcher};

/// Encapsulates logic for dispatching SMPC service requests.
#[derive(Debug)]
pub struct Dispatcher {}

impl Dispatcher {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl BatchDispatcher for Dispatcher {
    async fn dispatch_batch(&self, batch: Batch) {
        println!("TODO: dispatch request batch: {:?}", batch);
    }
}
