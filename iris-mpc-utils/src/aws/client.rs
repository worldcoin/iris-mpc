use std::time::Duration;

use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Builder, Client as SQSClient};
use thiserror::Error;

use iris_mpc_common::config::{ENV_PROD, ENV_STAGE};
use iris_mpc_common::helpers::sqs_s3_helper;

use super::config::AwsClientConfig;
use crate::{
    constants::N_PARTIES,
    misc::{log_error, log_info},
};

/// Component name for logging purposes.
const COMPONENT: &str = "State-AWS";

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct AwsClient {
    /// Associated configuration.
    config: AwsClientConfig,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    secrets_manager: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs: SQSClient,
}

// Network wide node AWS service clients.
pub type NetAwsClient = [AwsClient; N_PARTIES];

impl AwsClient {
    pub fn config(&self) -> &AwsClientConfig {
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

    pub fn new(config: AwsClientConfig) -> Self {
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

    pub async fn upload_to_s3(
        &self,
        bucket: &str,
        key: &str,
        payload: &[u8],
    ) -> Result<(), AwsClientError> {
        match sqs_s3_helper::upload_file_to_s3(bucket, key, self.s3().clone(), payload).await {
            Err(_) => Err(AwsClientError::S3UploadFailed),
            Ok(_) => Ok(()),
        }
    }
}

impl Clone for AwsClient {
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

#[derive(Error, Debug)]
pub enum AwsClientError {
    #[error("AWS S3 file upload error")]
    S3UploadFailed,
}

impl From<&AwsClientConfig> for S3Client {
    fn from(config: &AwsClientConfig) -> Self {
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

impl From<&AwsClientConfig> for SecretsManagerClient {
    fn from(config: &AwsClientConfig) -> Self {
        SecretsManagerClient::new(config.sdk())
    }
}

impl From<&AwsClientConfig> for SNSClient {
    fn from(config: &AwsClientConfig) -> Self {
        SNSClient::new(config.sdk())
    }
}

impl From<&AwsClientConfig> for SQSClient {
    fn from(config: &AwsClientConfig) -> Self {
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
    use super::super::{constants, AwsClient, AwsClientConfig};
    use crate::{constants::NODE_CONFIG_KIND_MAIN, fsys};

    fn assert_clients(clients: &AwsClient) {
        assert!(clients.s3().config().region().is_some());
        assert!(clients.secrets_manager().config().region().is_some());
        assert!(clients.sns().config().region().is_some());
        assert!(clients.sqs().config().region().is_some());
    }

    async fn create_clients() -> AwsClient {
        AwsClient::new(create_config().await)
    }

    async fn create_config() -> AwsClientConfig {
        let node_config = fsys::local::read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap();

        AwsClientConfig::new(
            node_config,
            constants::AWS_PUBLIC_KEY_BASE_URL.to_string(),
            constants::AWS_REQUESTS_BUCKET_NAME.to_string(),
            constants::AWS_REQUESTS_TOPIC_ARN.to_string(),
            constants::AWS_RESPONSE_QUEUE_URL.to_string(),
        )
        .await
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
