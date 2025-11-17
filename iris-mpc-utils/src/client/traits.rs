use async_trait::async_trait;

use super::{errors::ServiceClientError, types::RequestBatch};

/// Implemented by components which expose initialisation functions.
#[async_trait]
pub trait ComponentInitializer {
    async fn init(&mut self) -> Result<(), ServiceClientError>;
}

/// Implemented by components within batch processing pipeline.
#[async_trait]
pub trait ProcessRequestBatch {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ServiceClientError>;
}
