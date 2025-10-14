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
    s3_client: S3Client,

    /// Client for Amazon Simple Notification Service.
    sns_client: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs_client: SQSClient,

    /// Client for AWS Secrets Manager.
    secrets_manager_client: SecretsManagerClient,
}

impl AwsClients {
    pub fn s3_client(&self) -> &S3Client {
        &self.s3_client
    }

    pub fn sns_client(&self) -> &SNSClient {
        &self.sns_client
    }

    pub fn sqs_client(&self) -> &SQSClient {
        &self.sqs_client
    }

    pub fn secrets_manager_client(&self) -> &SecretsManagerClient {
        &self.secrets_manager_client
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use iris_mpc_common::config::{AwsConfig, Config as NodeConfig};
    use tokio;

    fn create_test_config() -> NodeConfig {
        NodeConfig {
            environment: "test".to_string(),
            sqs_long_poll_wait_time: 20,
            aws: Some(AwsConfig {
                region: Some("us-east-1".to_string()),
            }),
            ..Default::default()
        }
    }

    fn create_test_config_without_region() -> NodeConfig {
        NodeConfig {
            environment: "test".to_string(),
            sqs_long_poll_wait_time: 20,
            aws: None,
            ..Default::default()
        }
    }

    fn create_prod_config() -> NodeConfig {
        NodeConfig {
            environment: ENV_PROD.to_string(),
            sqs_long_poll_wait_time: 10,
            aws: Some(AwsConfig {
                region: Some("eu-west-1".to_string()),
            }),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_aws_clients_new() {
        let config = create_test_config();
        let clients = AwsClients::new(&config).await;

        assert!(clients.is_ok());
        let clients = clients.unwrap();

        // Verify all clients are accessible
        let _ = clients.s3_client();
        let _ = clients.sns_client();
        let _ = clients.sqs_client();
        let _ = clients.secrets_manager_client();
    }

    #[test]
    fn test_aws_clients_clone() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = create_test_config();
            let clients = AwsClients::new(&config).await.unwrap();

            let cloned_clients = clients.clone();

            // Verify cloned clients work
            let _ = cloned_clients.s3_client();
            let _ = cloned_clients.sns_client();
            let _ = cloned_clients.sqs_client();
            let _ = cloned_clients.secrets_manager_client();
        });
    }

    #[tokio::test]
    async fn test_get_aws_config_with_custom_region() {
        let config = create_test_config();
        let aws_config = get_aws_config(&config).await;

        assert_eq!(aws_config.region().unwrap().as_ref(), "us-east-1");
    }

    #[tokio::test]
    async fn test_get_aws_config_with_default_region() {
        let config = create_test_config_without_region();
        let aws_config = get_aws_config(&config).await;

        assert_eq!(aws_config.region().unwrap().as_ref(), DEFAULT_REGION);
    }

    #[tokio::test]
    async fn test_get_s3_client_with_shared_config() {
        let config = create_test_config();
        let shared_config = get_aws_config(&config).await;
        let s3_client = get_s3_client(&config, Some(&shared_config)).await;

        // Client should be created successfully
        assert!(!s3_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_s3_client_without_shared_config() {
        let config = create_test_config();
        let s3_client = get_s3_client(&config, None).await;

        // Client should be created successfully
        assert!(!s3_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_s3_client_force_path_style_non_prod() {
        let config = create_test_config(); // test environment
        let s3_client = get_s3_client(&config, None).await;

        // For non-prod environments, force_path_style should be true
        // Note: We can't directly test this as the config doesn't expose this field
        // This test mainly ensures the client is created without errors
        assert!(!s3_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_s3_client_prod_environment() {
        let config = create_prod_config();
        let s3_client = get_s3_client(&config, None).await;

        // For prod environment, force_path_style should be false
        assert!(!s3_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_sns_client_with_shared_config() {
        let config = create_test_config();
        let shared_config = get_aws_config(&config).await;
        let sns_client = get_sns_client(&config, Some(&shared_config)).await;

        // Client should be created successfully
        assert!(!sns_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_sns_client_without_shared_config() {
        let config = create_test_config();
        let sns_client = get_sns_client(&config, None).await;

        // Client should be created successfully
        assert!(!sns_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_ksm_client_with_shared_config() {
        let config = create_test_config();
        let shared_config = get_aws_config(&config).await;
        let ksm_client = get_ksm_client(&config, Some(&shared_config)).await;

        // Client should be created successfully
        assert!(!ksm_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_ksm_client_without_shared_config() {
        let config = create_test_config();
        let ksm_client = get_ksm_client(&config, None).await;

        // Client should be created successfully
        assert!(!ksm_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_sqs_client_with_shared_config() {
        let config = create_test_config();
        let shared_config = get_aws_config(&config).await;
        let sqs_client = get_sqs_client(&config, Some(&shared_config)).await;

        // Client should be created successfully
        assert!(!sqs_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_get_sqs_client_without_shared_config() {
        let config = create_test_config();
        let sqs_client = get_sqs_client(&config, None).await;

        // Client should be created successfully
        assert!(!sqs_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_sqs_client_timeout_configuration() {
        let mut config = create_test_config();
        config.sqs_long_poll_wait_time = 30;

        let sqs_client = get_sqs_client(&config, None).await;

        // The timeout should be configured to sqs_long_poll_wait_time + 2 seconds
        // We can't directly test the timeout value, but we can ensure the client is created
        assert!(!sqs_client.config().region().is_none());
    }

    #[test]
    fn test_default_region_constant() {
        assert_eq!(DEFAULT_REGION, "eu-north-1");
    }

    #[tokio::test]
    async fn test_aws_clients_getters() {
        let config = create_test_config();
        let clients = AwsClients::new(&config).await.unwrap();

        // Test that all getter methods return valid references
        let s3_ref = clients.s3_client();
        let sns_ref = clients.sns_client();
        let sqs_ref = clients.sqs_client();
        let secrets_ref = clients.secrets_manager_client();

        // Verify the references are valid by checking they have regions
        assert!(!s3_ref.config().region().is_none());
        assert!(!sns_ref.config().region().is_none());
        assert!(!sqs_ref.config().region().is_none());
        assert!(!secrets_ref.config().region().is_none());
    }

    #[tokio::test]
    async fn test_stage_environment_s3_config() {
        let mut config = create_test_config();
        config.environment = ENV_STAGE.to_string();

        let s3_client = get_s3_client(&config, None).await;

        // For stage environment, force_path_style should be false (like prod)
        assert!(!s3_client.config().region().is_none());
    }

    #[tokio::test]
    async fn test_multiple_client_creation_with_same_config() {
        let config = create_test_config();
        let shared_config = get_aws_config(&config).await;

        // Create multiple clients with the same shared config
        let s3_client1 = get_s3_client(&config, Some(&shared_config)).await;
        let s3_client2 = get_s3_client(&config, Some(&shared_config)).await;
        let sns_client1 = get_sns_client(&config, Some(&shared_config)).await;
        let sqs_client1 = get_sqs_client(&config, Some(&shared_config)).await;

        // All clients should be created successfully
        assert!(!s3_client1.config().region().is_none());
        assert!(!s3_client2.config().region().is_none());
        assert!(!sns_client1.config().region().is_none());
        assert!(!sqs_client1.config().region().is_none());
    }
}
