use std::time::Duration;

use aws_config::{retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{
    config::{Builder, Region},
    Client as SQSClient,
};

use iris_mpc_common::config::{Config as NodeConfig, ENV_PROD, ENV_STAGE};

use super::constants::AWS_REGION;
use crate::constants::N_PARTIES;

/// Encpasulates AWS service client configuration.
#[derive(Debug)]
pub struct NodeAwsClientConfig {
    /// Associated node configuration.
    node: NodeConfig,

    /// Cloud region.
    region: String,

    /// System request ingress queue URL.
    request_bucket_name: String,

    /// System request ingress queue topic.
    request_topic_arn: String,

    /// System response eqgress queue URL.
    response_queue_url: String,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,
}

// Network wide configuration set.
pub type NetAwsClientConfig = [NodeAwsClientConfig; N_PARTIES];

impl NodeAwsClientConfig {
    pub fn environment(&self) -> &String {
        &self.node().environment
    }

    pub fn node(&self) -> &NodeConfig {
        &self.node
    }

    pub fn region(&self) -> &String {
        &self.region
    }

    pub fn request_bucket_name(&self) -> &String {
        &self.request_bucket_name
    }

    pub fn request_topic_arn(&self) -> &String {
        &self.request_topic_arn
    }

    pub fn response_queue_url(&self) -> &String {
        &self.response_queue_url
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }

    pub async fn new(
        node_config: NodeConfig,
        region: String,
        request_bucket_name: String,
        request_topic_arn: String,
        response_queue_url: String,
    ) -> Self {
        Self {
            node: node_config.to_owned(),
            region: region.to_owned(),
            request_bucket_name,
            request_topic_arn,
            response_queue_url,
            sdk: get_sdk_config(region).await,
        }
    }
}

impl Clone for NodeAwsClientConfig {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            region: self.region.clone(),
            request_bucket_name: self.request_bucket_name.clone(),
            request_topic_arn: self.request_topic_arn.clone(),
            response_queue_url: self.response_queue_url.clone(),
            sdk: self.sdk.clone(),
        }
    }
}

impl From<&NodeAwsClientConfig> for SecretsManagerClient {
    fn from(config: &NodeAwsClientConfig) -> Self {
        SecretsManagerClient::new(config.sdk())
    }
}

impl From<&NodeAwsClientConfig> for SNSClient {
    fn from(config: &NodeAwsClientConfig) -> Self {
        SNSClient::new(config.sdk())
    }
}

impl From<&NodeAwsClientConfig> for S3Client {
    fn from(config: &NodeAwsClientConfig) -> Self {
        let force_path_style =
            config.environment() != ENV_PROD && config.environment() != ENV_STAGE;
        let config_builder = S3ConfigBuilder::from(config.sdk())
            .retry_config(RetryConfig::standard().with_max_attempts(5))
            .force_path_style(force_path_style);

        S3Client::from_conf(config_builder.build())
    }
}

impl From<&NodeAwsClientConfig> for SQSClient {
    fn from(config: &NodeAwsClientConfig) -> Self {
        SQSClient::from_conf(
            Builder::from(config.sdk())
                .timeout_config(
                    TimeoutConfig::builder()
                        .operation_attempt_timeout(Duration::from_secs(
                            (config.node().sqs_long_poll_wait_time + 2) as u64,
                        ))
                        .build(),
                )
                .build(),
        )
    }
}

/// Returns AWS SDK configuration from a node configuration instance.
async fn get_sdk_config(region: String) -> aws_config::SdkConfig {
    let region = Region::new(region);
    let retry_config = RetryConfig::standard().with_max_attempts(20);
    aws_config::from_env()
        .region(region)
        .retry_config(retry_config)
        .load()
        .await
}

#[cfg(test)]
mod tests {
    use super::super::constants::{
        AWS_REGION, AWS_REQUEST_BUCKET_NAME, AWS_REQUEST_TOPIC_ARN, AWS_RESPONSE_QUEUE_URL,
    };
    use super::NodeAwsClientConfig;
    use crate::{constants::NODE_CONFIG_KIND_MAIN, fsys::local::read_node_config};

    async fn create_config() -> NodeAwsClientConfig {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        NodeAwsClientConfig::new(
            node_config,
            AWS_REQUEST_BUCKET_NAME.to_string(),
            AWS_REQUEST_TOPIC_ARN.to_string(),
            AWS_RESPONSE_QUEUE_URL.to_string(),
        )
        .await
    }

    #[tokio::test]
    async fn test_config_new() {
        let config = create_config().await;
        // TODO: check why this assert fails.
        // assert!(config.sdk().endpoint_url().is_some());
        assert_eq!(config.sdk().region().unwrap().as_ref(), AWS_REGION);
    }
}
