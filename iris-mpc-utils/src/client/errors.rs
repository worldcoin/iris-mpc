use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum ServiceClientError {
    #[error("Component initialisation error: {0}")]
    ComponentInitialisationError(String),

    #[error("Enqueue uniqueness request error: {0}")]
    EnqueueUniquenessRequestError(String),

    #[error("An AWS service error has occured: {0}")]
    AwsServiceError(String),

    #[error("Request type is unsupported: {0}")]
    UnsupportedRequestType(String),
}
