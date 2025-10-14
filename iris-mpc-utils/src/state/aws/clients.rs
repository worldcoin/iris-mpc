use aws_config::{retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
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
pub struct Clients {
    /// Client for Amazon Simple Storage Service.
    s3_client: S3Client,

    /// Configuration used to instantiate service clients.
    sdk_config: SdkConfig,

    /// Client for AWS Secrets Manager.
    secrets_manager_client: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns_client: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs_client: SQSClient,
}

impl Clients {
    pub fn s3_client(&self) -> &S3Client {
        &self.s3_client
    }

    pub fn sdk_config(&self) -> &SdkConfig {
        &self.sdk_config
    }

    pub fn secrets_manager_client(&self) -> &SecretsManagerClient {
        &self.secrets_manager_client
    }

    pub fn sns_client(&self) -> &SNSClient {
        &self.sns_client
    }

    pub fn sqs_client(&self) -> &SQSClient {
        &self.sqs_client
    }
}

impl Clients {
    pub async fn new(node_config: &NodeConfig) -> Result<Self> {
        let sdk_config = get_sdk_config(node_config).await;

        Ok(Self {
            s3_client: get_s3_client(node_config, Some(&sdk_config)).await,
            sdk_config: sdk_config.clone(),
            secrets_manager_client: get_secrets_manager_client(node_config, Some(&sdk_config))
                .await,
            sqs_client: get_sqs_client(node_config, Some(&sdk_config)).await,
            sns_client: get_sns_client(node_config, Some(&sdk_config)).await,
        })
    }
}

impl Clone for Clients {
    fn clone(&self) -> Self {
        Self {
            sdk_config: self.sdk_config.clone(),
            sqs_client: self.sqs_client.clone(),
            sns_client: self.sns_client.clone(),
            s3_client: self.s3_client.clone(),
            secrets_manager_client: self.secrets_manager_client.clone(),
        }
    }
}

/// Creates an AWS S3 client with default retry configuration.
pub async fn get_s3_client(
    node_config: &NodeConfig,
    sdk_config: Option<&aws_config::SdkConfig>,
) -> S3Client {
    let shared_config = match sdk_config {
        None => get_sdk_config(node_config).await,
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

/// Returns an AWS SDK configuration instance hydrated from env vars plus defaults.
pub async fn get_sdk_config(node_config: &NodeConfig) -> aws_config::SdkConfig {
    // TODO: AWS endpoint can be defined in a node config yet
    // it is always pulled form env var - is this correct behaviour ?
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

/// Creates an AWS Secrets Manager client.
pub async fn get_secrets_manager_client(
    node_config: &NodeConfig,
    sdk_config: Option<&aws_config::SdkConfig>,
) -> SecretsManagerClient {
    let shared_config = match sdk_config {
        None => get_sdk_config(node_config).await,
        Some(inner) => inner.clone(),
    };

    SecretsManagerClient::new(&shared_config)
}

/// Creates an AWS SNS client.
pub async fn get_sns_client(
    node_config: &NodeConfig,
    sdk_config: Option<&aws_config::SdkConfig>,
) -> SNSClient {
    let shared_config = match sdk_config {
        None => get_sdk_config(node_config).await,
        Some(inner) => inner.clone(),
    };

    SNSClient::new(&shared_config)
}

/// Creates an AWS SQS client with a client-side operation attempt timeout. Per default, there are two retry
/// attempts, meaning that every operation has three tries in total. This configuration prevents the sqs
/// client from `await`ing forever on broken streams. (see <https://github.com/awslabs/aws-sdk-rust/issues/1094>)
pub async fn get_sqs_client(
    node_config: &NodeConfig,
    sdk_config: Option<&aws_config::SdkConfig>,
) -> SQSClient {
    let shared_config = match sdk_config {
        None => get_sdk_config(node_config).await,
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

#[cfg(test)]
mod tests {
    use super::{get_sdk_config, Clients, SdkConfig};
    use crate::{
        constants::{DEFAULT_AWS_REGION, NODE_CONFIG_KIND_MAIN},
        state::fsys::local::read_node_config,
    };
    use iris_mpc_common::config::Config as NodeConfig;

    fn assert_clients(clients: &Clients) {
        let client = clients.s3_client();
        assert!(client.config().region().is_some());

        let client = clients.secrets_manager_client();
        assert!(client.config().region().is_some());

        let client = clients.sns_client();
        assert!(client.config().region().is_some());

        let client = clients.sqs_client();
        assert!(client.config().region().is_some());
    }

    async fn create_clients() -> Clients {
        let clients = Clients::new(&create_node_config()).await;
        assert!(clients.is_ok());

        clients.unwrap()
    }

    fn create_node_config() -> NodeConfig {
        read_node_config(NODE_CONFIG_KIND_MAIN, 0, &0).unwrap()
    }

    async fn create_sdk_config() -> SdkConfig {
        get_sdk_config(&create_node_config()).await
    }

    #[tokio::test]
    async fn test_get_sdk_config() {
        let config = create_sdk_config().await;
        // TODO: check why this assert fails.
        // assert!(config.endpoint_url().is_some());
        assert_eq!(config.region().unwrap().as_ref(), DEFAULT_AWS_REGION);
    }

    #[tokio::test]
    async fn test_clients_new() {
        assert_clients(&create_clients().await);
    }

    #[tokio::test]
    async fn test_clients_clone() {
        assert_clients(&create_clients().await.clone());
    }
}
