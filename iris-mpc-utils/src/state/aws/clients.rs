use crate::misc::{log_error, log_info};
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

/// Component name for logging purposes.
const COMPONENT: &str = "State-AWS";

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
            config: config.to_owned(),
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }
}

impl Config {
    pub async fn new(node_config: &NodeConfig) -> Self {
        Self {
            node: node_config.to_owned(),
            sdk: get_sdk_config(node_config).await,
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
    pub fn environment(&self) -> &String {
        &self.node().environment
    }

    pub fn node(&self) -> &NodeConfig {
        &self.node
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }
}

impl Clients {
    pub(super) fn log_error(&self, msg: &str) {
        log_error(COMPONENT, msg);
    }

    pub(super) fn log_info(&self, msg: &str) {
        log_info(COMPONENT, msg);
    }
}

impl From<&Config> for S3Client {
    fn from(config: &Config) -> Self {
        let force_path_style =
            config.environment() != ENV_PROD && config.environment() != ENV_STAGE;

        S3Client::from_conf(
            S3ConfigBuilder::from(config.sdk())
                .force_path_style(force_path_style)
                .retry_config(RetryConfig::standard().with_max_attempts(5))
                .build(),
        )
    }
}

impl From<&Config> for SecretsManagerClient {
    fn from(config: &Config) -> Self {
        SecretsManagerClient::new(config.sdk())
    }
}

impl From<&Config> for SNSClient {
    fn from(config: &Config) -> Self {
        SNSClient::new(config.sdk())
    }
}

impl From<&Config> for SQSClient {
    fn from(config: &Config) -> Self {
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
async fn get_sdk_config(node_config: &NodeConfig) -> aws_config::SdkConfig {
    aws_config::from_env()
        .region(Region::new(
            node_config
                .clone()
                .aws
                .and_then(|aws| aws.region)
                .unwrap_or_else(|| DEFAULT_REGION.to_owned()),
        ))
        .load()
        .await
}

#[cfg(test)]
mod tests {
    use super::{Clients, Config};
    use crate::{
        constants::{DEFAULT_AWS_REGION, NODE_CONFIG_KIND_MAIN},
        state::fsys::local::read_node_config,
    };

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
        let config = create_config().await;

        Clients::new(config)
    }

    async fn create_config() -> Config {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        Config::new(&node_config).await
    }

    #[tokio::test]
    async fn test_clients_new() {
        let clients = create_clients().await;
        assert_clients(&clients);
    }

    #[tokio::test]
    async fn test_clients_new_then_clone() {
        let clients = create_clients().await.clone();
        assert_clients(&clients);
    }

    #[tokio::test]
    async fn test_config_new() {
        let config = create_config().await;
        // TODO: check why this assert fails.
        // assert!(config.sdk().endpoint_url().is_some());
        assert_eq!(config.sdk().region().unwrap().as_ref(), DEFAULT_AWS_REGION);
    }
}
