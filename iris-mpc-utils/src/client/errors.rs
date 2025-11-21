use crate::aws::AwsClientError;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum ServiceClientError {
    #[error("An AWS service error has occured: {0}")]
    AwsServiceError(#[from] AwsClientError),

    #[error("Component initialisation error: {0}")]
    ComponentInitialisationError(String),

    #[error("Enqueue uniqueness request error: {0}")]
    EnqueueUniquenessRequestError(String),

    #[error("Service client initialisation error: {0}")]
    InitialisationError(String),

    #[error("Enqueue uniqueness pre-enqueue setup error: {0}")]
    PrequeueUniquenessRequestError(String),

    #[error("Request type is unsupported: {0}")]
    UnsupportedRequestType(String),
}
