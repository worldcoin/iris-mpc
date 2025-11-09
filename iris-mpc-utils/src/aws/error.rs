use thiserror::Error;

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum AwsClientError {
    #[error("AWS SQS purge queue error: {}", .error)]
    SqsPurgeQueueError { error: String },

    #[error("AWS SNS publish error: {}", .error)]
    SnsPublishError { error: String },

    #[error("AWS S3 upload error: key={}: error={}", .key, .error)]
    S3UploadError { key: String, error: String },

    #[error("Download encryption keys error: {}", .error)]
    EncryptionKeysDownloadError { error: String },

    #[error("Iris shares encrypt and upload error: {}", .error)]
    IrisSharesEncryptAndUploadError { error: String },

    #[error("Iris deletions upload error: {}", .error)]
    IrisDeletionsUploadError { error: String },
}
