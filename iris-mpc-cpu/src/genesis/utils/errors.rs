use thiserror::Error;

// Encpasulates a non-exhaustive set of errors raised during indexation.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum IndexationError {
    #[error("Failed to download AWS S3 object")]
    AwsS3ObjectDownload,

    #[error("Failed to deserialize AWS S3 object")]
    AwsS3ObjectDeserialize,

    #[error("Failed to begin indexation correctly")]
    BeginIndexation,

    #[error("Failed to spawn Hawk actor")]
    HawkActor,

    #[error("Failed to connect to PostgreSQL dB")]
    PostgresConnection,

    #[error("Failed to fetch a batch of Iris data from PostgreSQL dB")]
    PostgresFetchIrisBatch,

    #[error("Failed to fetch Iris data from PostgreSQL dB")]
    PostgresFetchIrisById,
}
