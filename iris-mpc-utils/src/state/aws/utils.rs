use aws_config::{self, retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
use aws_sdk_s3::{
    config::{Builder as S3_ConfigBuilder, Region as AWS_Region},
    primitives::ByteStream as S3_ByteStream,
    Client as S3_Client,
};
use aws_sdk_sqs::{config::Builder as SQS_ConfigBuilder, Client as SQS_Client};
use eyre::Result;
use iris_mpc_common::config::{Config as NodeConfig, ENV_PROD, ENV_STAGE};
use std::time::Duration;

/// Default AWS region.
const DEFAULT_AWS_REGION: &str = "eu-north-1";

/// Returns an S3 client with retry configuration.
#[allow(dead_code)]
pub async fn get_s3_client(config: &NodeConfig) -> Result<S3_Client> {
    let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
    let retry_config = RetryConfig::standard().with_max_attempts(5);
    let shared_config = get_shared_config(config).await?;
    let s3_config = S3_ConfigBuilder::from(&shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    Ok(S3_Client::from_conf(s3_config))
}

/// Returns an SQS client with default retry configuration.
#[allow(dead_code)]
pub async fn get_sqs_client(config: &NodeConfig) -> Result<SQS_Client> {
    let shared_config = get_shared_config(config).await?;
    let timeout_threshold = Duration::from_secs((config.sqs_long_poll_wait_time + 2) as u64);
    let sqs_config = SQS_ConfigBuilder::from(&shared_config)
        .timeout_config(
            TimeoutConfig::builder()
                .operation_attempt_timeout(timeout_threshold)
                .build(),
        )
        .build();

    Ok(SQS_Client::from_conf(sqs_config))
}

/// Returns base AWS SDK configuration used across clients.
async fn get_shared_config(config: &NodeConfig) -> Result<SdkConfig> {
    let region = AWS_Region::new(
        config
            .to_owned()
            .aws
            .and_then(|aws| aws.region)
            .unwrap_or_else(|| DEFAULT_AWS_REGION.to_owned()),
    );

    Ok(aws_config::from_env().region(region).load().await)
}

/// Uploads data to a node's S3 bucket.
#[allow(dead_code)]
pub async fn upload_to_s3(
    client: &S3_Client,
    bucket_name: &str,
    object_key: &str,
    object_data: S3_ByteStream,
) -> Result<()> {
    client
        .put_object()
        .bucket(bucket_name)
        .key(object_key)
        .body(object_data)
        .send()
        .await?;

    Ok(())
}

/// Uploads data to a node's S3 bucket.
#[allow(dead_code)]
pub async fn upload_to_s3_of_node(
    config: &NodeConfig,
    bucket_name: &str,
    object_key: &str,
    object_data: S3_ByteStream,
) -> Result<()> {
    let client = get_s3_client(config).await?;

    upload_to_s3(&client, bucket_name, object_key, object_data).await
}

#[cfg(test)]
mod test {
    use super::get_s3_client;
    use crate::{constants::NODE_CONFIG_KIND_GENESIS, state::fsys, types::NetConfig};

    fn get_net_config() -> NetConfig {
        fsys::local::read_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap()
    }

    #[tokio::test]
    async fn test_get_s3_client() {
        for (party_idx, cfg) in get_net_config().iter().enumerate() {
            match get_s3_client(cfg).await {
                Ok(_) => (),
                Err(err) => panic!(
                    "Failed to get S3 client: Party IDX={} :: Err={}",
                    party_idx, err
                ),
            };
        }
    }
}
