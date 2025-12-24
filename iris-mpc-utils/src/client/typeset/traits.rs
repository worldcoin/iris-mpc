use async_trait::async_trait;

use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::{data::RequestBatch, errors::ServiceClientError};
use crate::types::IrisCodeAndMaskShares;

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

/// Implemented by Iris shares generators.
#[async_trait]
pub trait GenerateShares {
    async fn generate(&mut self) -> BothEyes<IrisCodeAndMaskShares>;
}
