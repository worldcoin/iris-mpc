use async_trait::async_trait;

use super::{errors::ServiceClientError, types::RequestBatch};

/// Implemented by components within batch processing pipeline.
#[async_trait]
pub trait RequestBatchProcesser {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ServiceClientError>;
}
