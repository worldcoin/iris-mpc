use thiserror::Error;

// Encpasulates a non-exhaustive set of errors raised during indexation.
#[derive(Error, Debug)]
pub enum IndexationError {
    #[error("Failed to load AWS configuration")]
    AwsConfigurationError,

    #[error("Failed to download resource from AWS S3")]
    AwsS3DownloadError,

    #[error("Failed to begin indexation correctly")]
    BeginIndexationError,

    #[error("Failed to spawn Hawk actor")]
    HawkActorError,

    #[error("Failed to connect to PostgreSQL dB")]
    PostgresConnectionError,

    #[error("Failed to fetch Iris data from PostgreSQL dB")]
    PostgresFetchIrisByIdError,
}
