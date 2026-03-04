use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum AwsClientError {
    #[error("Download public key set error: {0}")]
    PublicKeysetDownloadError(String),

    #[error("Iris shares encrypt and upload error: {0}")]
    IrisSharesEncryptAndUploadError(String),

    #[error("Iris deletions upload error: {0}")]
    IrisDeletionsUploadError(String),

    #[error("AWS S3 upload error: key={0}: error={0}")]
    S3UploadError(String, String),

    #[error("AWS SNS publish error: {0}")]
    SnsPublishError(String),

    #[error("AWS SQS delete message from queue error: {0}")]
    SqsDeleteMessageError(String),

    #[error("AWS SQS purge queue error: {0}")]
    SqsPurgeQueueError(String),

    #[error("AWS SQS receive message from queue error: {0}")]
    SqsReceiveMessageError(String),
}
