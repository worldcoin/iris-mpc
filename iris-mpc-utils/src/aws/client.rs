use std::time::Duration;

use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Builder, Client as SQSClient};

use iris_mpc_common::config::{ENV_PROD, ENV_STAGE};

use super::config::NodeAwsConfig;
use crate::misc::{log_error, log_info};

/// Component name for logging purposes.
const COMPONENT: &str = "State-AWS";

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct NodeAwsClients {
    /// Associated configuration.
    config: NodeAwsConfig,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    secrets_manager: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs: SQSClient,
}

impl NodeAwsClients {
    pub fn config(&self) -> &NodeAwsConfig {
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

    pub fn new(config: NodeAwsConfig) -> Self {
        Self {
            config: config.to_owned(),
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }

    pub(super) fn log_error(&self, msg: &str) {
        log_error(COMPONENT, msg);
    }

    pub(super) fn log_info(&self, msg: &str) {
        log_info(COMPONENT, msg);
    }
}

impl Clone for NodeAwsClients {
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

impl From<&NodeAwsConfig> for S3Client {
    fn from(config: &NodeAwsConfig) -> Self {
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

impl From<&NodeAwsConfig> for SecretsManagerClient {
    fn from(config: &NodeAwsConfig) -> Self {
        SecretsManagerClient::new(config.sdk())
    }
}

impl From<&NodeAwsConfig> for SNSClient {
    fn from(config: &NodeAwsConfig) -> Self {
        SNSClient::new(config.sdk())
    }
}

impl From<&NodeAwsConfig> for SQSClient {
    fn from(config: &NodeAwsConfig) -> Self {
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

#[cfg(test)]
mod tests {
    use super::{super::config::NodeAwsConfig, NodeAwsClients};
    use crate::{constants::NODE_CONFIG_KIND_MAIN, fsys::local::read_node_config};

    fn assert_clients(clients: &NodeAwsClients) {
        let client = clients.s3();
        assert!(client.config().region().is_some());

        let client = clients.secrets_manager();
        assert!(client.config().region().is_some());

        let client = clients.sns();
        assert!(client.config().region().is_some());

        let client = clients.sqs();
        assert!(client.config().region().is_some());
    }

    async fn create_clients() -> NodeAwsClients {
        let config = create_config().await;

        NodeAwsClients::new(config)
    }

    async fn create_config() -> NodeAwsConfig {
        let node_config = read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        NodeAwsConfig::new(node_config).await
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
}
