use crate::aws::AwsClientError;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum ServiceClientError {
    #[error("An AWS service error has occured: {0}")]
    AwsServiceError(#[from] AwsClientError),

    #[error("Enqueue request error: {0}")]
    EnqueueRequestError(String),

    #[error("Initialisation error: {0}")]
    InitialisationError(String),

    #[error("Response error: {0}")]
    ResponseError(String),
}
