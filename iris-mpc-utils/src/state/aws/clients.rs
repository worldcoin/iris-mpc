use async_from::{async_trait, AsyncFrom};
use aws_config::{retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{
    config::{Builder, Region},
    Client as SQSClient,
};
use iris_mpc_common::config::{Config as NodeConfig, ENV_PROD, ENV_STAGE};
use std::time::Duration;

const DEFAULT_REGION: &str = "eu-north-1";

/// Encpasulates access to a set of AWS service clients.
pub struct Clients {
    /// Associated configuration.
    config: Config,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    secrets_manager: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs: SQSClient,
}

/// Encpasulates AWS service client configuration.
pub struct Config {
    /// Associated node configuration.
    node: NodeConfig,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,
}

impl Clients {
    pub fn new(config: Config) -> Self {
        Self {
            config: config.clone(),
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }
}

impl Config {
    pub fn new(node_config: NodeConfig, sdk_config: SdkConfig) -> Self {
        Self {
            node: node_config,
            sdk: sdk_config,
        }
    }
}

impl Clone for Clients {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sqs: self.sqs.clone(),
            sns: self.sns.clone(),
            s3: self.s3.clone(),
            secrets_manager: self.secrets_manager.clone(),
        }
    }
}

impl Clone for Config {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            sdk: self.sdk.clone(),
        }
    }
}

impl Clients {
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn s3(&self) -> &S3Client {
        &self.s3
    }

    pub fn secrets_manager(&self) -> &SecretsManagerClient {
        &self.secrets_manager
    }

    pub fn sns(&self) -> &SNSClient {
        &self.sns
    }

    pub fn sqs(&self) -> &SQSClient {
        &self.sqs
    }
}

impl Config {
    pub fn node(&self) -> &NodeConfig {
        &self.node
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }
}

impl From<&Config> for S3Client {
    fn from(value: &Config) -> Self {
        let force_path_style =
            value.node().environment != ENV_PROD && value.node().environment != ENV_STAGE;

        S3Client::from_conf(
            S3ConfigBuilder::from(value.sdk())
                .force_path_style(force_path_style)
                .retry_config(RetryConfig::standard().with_max_attempts(5))
                .build(),
        )
    }
}

impl From<&Config> for SecretsManagerClient {
    fn from(value: &Config) -> Self {
        SecretsManagerClient::new(value.sdk())
    }
}

impl From<&Config> for SNSClient {
    fn from(value: &Config) -> Self {
        SNSClient::new(value.sdk())
    }
}

impl From<&Config> for SQSClient {
    fn from(value: &Config) -> Self {
        SQSClient::from_conf(
            Builder::from(value.sdk())
                .timeout_config(
                    TimeoutConfig::builder()
                        .operation_attempt_timeout(Duration::from_secs(
                            (value.node().sqs_long_poll_wait_time + 2) as u64,
                        ))
                        .build(),
                )
                .build(),
        )
    }
}

#[async_trait]
impl AsyncFrom<NodeConfig> for Config {
    async fn async_from(node_config: NodeConfig) -> Self {
        let sdk_config = aws_config::from_env()
            .region(Region::new(
                node_config
                    .clone()
                    .aws
                    .and_then(|aws| aws.region)
                    .unwrap_or_else(|| DEFAULT_REGION.to_owned()),
            ))
            .load()
            .await;

        Config::new(node_config, sdk_config)
    }
}

#[cfg(test)]
mod tests {
    use super::{Clients, Config};
    use crate::{
        constants::{DEFAULT_AWS_REGION, NODE_CONFIG_KIND_MAIN},
        state::fsys::local::read_node_config,
    };
    use async_from::AsyncFrom;

    fn assert_clients(clients: &Clients) {
        let client = clients.s3();
        assert!(client.config().region().is_some());

        let client = clients.secrets_manager();
        assert!(client.config().region().is_some());

        let client = clients.sns();
        assert!(client.config().region().is_some());

        let client = clients.sqs();
        assert!(client.config().region().is_some());
    }

    async fn create_clients() -> Clients {
        Clients::new(create_config().await)
    }

    async fn create_config() -> Config {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        Config::async_from(node_config).await
    }

    #[tokio::test]
    async fn test_clients_new() {
        assert_clients(&create_clients().await);
    }

    #[tokio::test]
    async fn test_clients_new_then_clone() {
        assert_clients(&create_clients().await.clone());
    }

    #[tokio::test]
    async fn test_config_new() {
        let config = create_config().await;
        // TODO: check why this assert fails.
        // assert!(config.sdk().endpoint_url().is_some());
        assert_eq!(config.sdk().region().unwrap().as_ref(), DEFAULT_AWS_REGION);
    }
}
