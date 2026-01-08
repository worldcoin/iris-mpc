use async_trait::async_trait;

use super::{data::RequestBatch, errors::ServiceClientError};

/// Implemented by components which expose initialisation functions.
#[async_trait]
pub(crate) trait Initialize {
    async fn init(&mut self) -> Result<(), ServiceClientError>;
}

/// Implemented by components within batch processing pipeline.
#[async_trait]
pub(crate) trait ProcessRequestBatch {
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ServiceClientError>;
}
