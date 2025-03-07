use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::{
    config::{Builder as S3ConfigBuilder, StalledStreamProtectionConfig},
    Client as S3Client,
};
use std::time::Duration;

/// Creates an S3 client with retry configuration
pub fn create_s3_client(shared_config: &aws_config::SdkConfig, force_path_style: bool) -> S3Client {
    let retry_config = RetryConfig::standard().with_max_attempts(5);

    let s3_config = S3ConfigBuilder::from(shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    S3Client::from_conf(s3_config)
}

/// Creates an S3 client specifically for database chunks with additional
/// configuration
pub fn create_db_chunks_s3_client(
    shared_config: &aws_config::SdkConfig,
    force_path_style: bool,
) -> S3Client {
    let retry_config = RetryConfig::standard().with_max_attempts(5);

    // Increase S3 connect timeouts to 10s
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(10))
        .build();

    let db_chunks_s3_config = S3ConfigBuilder::from(shared_config)
        // disable stalled stream protection to avoid panics during s3 import
        .stalled_stream_protection(StalledStreamProtectionConfig::disabled())
        .retry_config(retry_config)
        .timeout_config(timeout_config)
        .force_path_style(force_path_style)
        .build();

    S3Client::from_conf(db_chunks_s3_config)
}
