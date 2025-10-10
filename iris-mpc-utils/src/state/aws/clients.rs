use aws_config::{retry::RetryConfig, timeout::TimeoutConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{
    config::{Builder, Region},
    Client as SQSClient,
};
use eyre::Result;
use iris_mpc_common::config::{Config as NodeConfig, ENV_PROD, ENV_STAGE};
use std::time::Duration;

const DEFAULT_REGION: &str = "eu-north-1";

/// Encpasulates access to a set of AWS service clients.
pub struct AwsClients {
    /// Client for Amazon Simple Storage Service.
    pub s3_client: S3Client,

    /// Client for Amazon Simple Notification Service.
    pub sns_client: SNSClient,

    /// Client for Amazon Simple Queue Service.
    pub sqs_client: SQSClient,

    /// Client for AWS Secrets Manager.
    pub secrets_manager_client: SecretsManagerClient,
}

impl AwsClients {
    pub async fn new(config: &NodeConfig) -> Result<Self> {
        let shared_config = get_aws_config(config).await;

        Ok(Self {
            sqs_client: get_sqs_client(config, Some(&shared_config)).await,
            sns_client: get_sns_client(config, Some(&shared_config)).await,
            s3_client: get_s3_client(config, Some(&shared_config)).await,
            secrets_manager_client: get_ksm_client(config, Some(&shared_config)).await,
        })
    }
}

impl Clone for AwsClients {
    fn clone(&self) -> Self {
        Self {
            sqs_client: self.sqs_client.clone(),
            sns_client: self.sns_client.clone(),
            s3_client: self.s3_client.clone(),
            secrets_manager_client: self.secrets_manager_client.clone(),
        }
    }
}

/// Returns an AWS configuration instance hydrated from env vars plus defaults.
pub async fn get_aws_config(node_config: &NodeConfig) -> aws_config::SdkConfig {
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

/// Creates an AWS S3 client with default retry configuration.
pub async fn get_s3_client(
    node_config: &NodeConfig,
    shared_config: Option<&aws_config::SdkConfig>,
) -> S3Client {
    let shared_config = match shared_config {
        None => get_aws_config(node_config).await,
        Some(inner) => inner.clone(),
    };
    let force_path_style =
        node_config.environment != ENV_PROD && node_config.environment != ENV_STAGE;

    S3Client::from_conf(
        S3ConfigBuilder::from(&shared_config)
            .force_path_style(force_path_style)
            .retry_config(RetryConfig::standard().with_max_attempts(5))
            .build(),
    )
}

/// Creates an AWS SNS client.
pub async fn get_sns_client(
    node_config: &NodeConfig,
    shared_config: Option<&aws_config::SdkConfig>,
) -> SNSClient {
    let shared_config = match shared_config {
        None => get_aws_config(node_config).await,
        Some(inner) => inner.clone(),
    };

    SNSClient::new(&shared_config)
}

/// Creates an AWS KSM client.
pub async fn get_ksm_client(
    node_config: &NodeConfig,
    shared_config: Option<&aws_config::SdkConfig>,
) -> SecretsManagerClient {
    let shared_config = match shared_config {
        None => get_aws_config(node_config).await,
        Some(inner) => inner.clone(),
    };

    SecretsManagerClient::new(&shared_config)
}

/// Creates an AWS SQS client with a client-side operation attempt timeout. Per default, there are two retry
/// attempts, meaning that every operation has three tries in total. This configuration prevents the sqs
/// client from `await`ing forever on broken streams. (see <https://github.com/awslabs/aws-sdk-rust/issues/1094>)
pub async fn get_sqs_client(
    node_config: &NodeConfig,
    shared_config: Option<&aws_config::SdkConfig>,
) -> SQSClient {
    let shared_config = match shared_config {
        None => get_aws_config(node_config).await,
        Some(inner) => inner.clone(),
    };

    SQSClient::from_conf(
        Builder::from(&shared_config)
            .timeout_config(
                TimeoutConfig::builder()
                    .operation_attempt_timeout(Duration::from_secs(
                        (node_config.sqs_long_poll_wait_time + 2) as u64,
                    ))
                    .build(),
            )
            .build(),
    )
}
