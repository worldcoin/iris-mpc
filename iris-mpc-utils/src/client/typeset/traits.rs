use async_trait::async_trait;

use super::{data::RequestBatch, errors::ClientError};

/// Implemented by components which expose initialisation functions.
#[async_trait]
pub trait Initialize {
    async fn init(&mut self) -> Result<(), ClientError>;
}

/// Implemented by components within batch processing pipeline.
#[async_trait]
pub trait ProcessRequestBatch {
    async fn process_batch(&mut self, batch: &RequestBatch) -> Result<(), ClientError>;
}
