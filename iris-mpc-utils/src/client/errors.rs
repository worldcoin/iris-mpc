use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum ServiceClientError {
    #[error("Component initialisation error: {}", .error)]
    ComponentInitialisationError { error: String },
}
