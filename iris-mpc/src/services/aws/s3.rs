use aws_config::retry::RetryConfig;
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, config::Region, Client as S3Client};

/// Creates an S3 client with retry configuration
///
/// # Arguments
/// * `shared_config` - Base AWS SDK configuration to build from
/// * `force_path_style` - Whether to use path-style S3 URLs
/// * `region_override` - Optional region to override the shared config's region
pub fn create_s3_client(
    shared_config: &aws_config::SdkConfig,
    force_path_style: bool,
    region_override: Option<Region>,
) -> S3Client {
    let retry_config = RetryConfig::standard().with_max_attempts(5);

    let mut builder = S3ConfigBuilder::from(shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone());

    if let Some(region) = region_override {
        builder = builder.region(region);
    }

    let s3_config = builder.build();

    S3Client::from_conf(s3_config)
}
