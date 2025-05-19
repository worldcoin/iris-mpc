use super::logger;
use eyre::{bail, Result};
use thiserror::Error;

// Encpasulates a non-exhaustive set of errors raised during indexation.
#[derive(Error, Debug)]
pub enum IndexationError {
    #[error("Failed to download AWS S3 object")]
    AwsS3ObjectDownload,

    #[error("Failed to deserialize AWS S3 object")]
    AwsS3ObjectDeserialize,

    #[error("Current height of indexation exceeds maximum allowed")]
    IndexationHeightMismatch,

    #[error("Failed to fetch Iris batch from PostgreSQL dB: {0}")]
    PostgresFetchIrisBatch(String),

    #[error("Failed to fetch Iris data from PostgreSQL dB")]
    PostgresFetchIrisById,
}

// Helper: handles error.
pub fn handle_error(msg: String) -> Result<()> {
    let msg = logger::log_error("Server", msg);

    bail!(msg);
}
