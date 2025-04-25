use thiserror::Error;

// Encpasulates a non-exhaustive set of errors raised during indexation.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum IndexationError {
    #[error("Failed to load AWS configuration")]
    AwsConfiguration,

    #[error("Failed to download resource from AWS S3")]
    AwsS3Download,

    #[error("Failed to begin indexation correctly")]
    BeginIndexation,

    #[error("Failed to spawn Hawk actor")]
    HawkActor,

    #[error("Failed to connect to PostgreSQL dB")]
    PostgresConnection,

    #[error("Failed to fetch Iris data from PostgreSQL dB")]
    PostgresFetchIrisById,
}
