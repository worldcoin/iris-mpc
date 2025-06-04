use iris_mpc_common::IrisSerialId;
use thiserror::Error;

// Encpasulates a non-exhaustive set of errors raised during indexation.
#[derive(Error, Debug)]
pub enum IndexationError {
    #[error("AWS RDS cluster URL is invalid ... check dB configuration")]
    AwsRdsInvalidClusterURL,

    #[error("AWS RDS cluster ID not found ... snapshotting failed")]
    AwsRdsClusterIdNotFound,

    #[error("AWS RDS cluster URL is invalid ... check dB configuration")]
    AwsRdsGetClusterURLs,

    #[error("AWS RDS cluster snapshotting failed: {0}")]
    AwsRdsCreateSnapshotFailure(String),

    #[error("Failed to download AWS S3 object")]
    AwsS3ObjectDownload,

    #[error("Failed to deserialize AWS S3 object")]
    AwsS3ObjectDeserialize,

    #[error("Missing CPU db configuration")]
    DbConfigError,

    #[error("Current height of indexation exceeds maximum allowed")]
    IndexationHeightMismatch,

    #[error("Failed to fetch Iris with given serial ID: {0}")]
    MissingSerialId(IrisSerialId),

    #[error("Failed to fetch Iris batch from PostgreSQL dB: {0}")]
    PostgresFetchIrisBatch(String),

    #[error("Failed to fetch Modification batch from PostgreSQL dB: {0}")]
    PostgresFetchModificationBatch(String),

    #[error("Failed to persist genesis Graph indexation state to PostgreSQL dB: {0}")]
    PostgresPersistIndexationState(String),
}
