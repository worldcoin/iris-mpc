use aws_sdk_s3::{config::Region, Client as S3Client};

/// Creates an S3 client with the shared retry + per-attempt-timeout policy.
///
/// Delegates to [`iris_mpc_cpu::graph_checkpoint::create_s3_client`] so hawk,
/// genesis and the sidecar construct S3 clients identically.
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
    iris_mpc_cpu::graph_checkpoint::create_s3_client(
        shared_config,
        force_path_style,
        region_override,
    )
}
