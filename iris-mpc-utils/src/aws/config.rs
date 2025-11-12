use std::time::Duration;

use aws_config::{retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Builder, Client as SQSClient};

use iris_mpc_common::config::{ENV_PROD, ENV_STAGE};

/// Encpasulates AWS service client configuration.
#[derive(Debug)]
pub struct AwsClientConfig {
    /// Execution environment.
    environment: String,

    /// System request ingress queue URL.
    s3_request_bucket_name: String,

    /// Associated AWS SDK configuration.
    sdk: SdkConfig,

    /// System request ingress queue topic.
    sns_request_topic_arn: String,

    /// Polling interval between AWS SQS interactions.
    sqs_long_poll_wait_time: usize,

    /// System response eqgress queue URL.
    sqs_response_queue_url: String,
}

impl AwsClientConfig {
    pub fn environment(&self) -> &String {
        &self.environment
    }

    pub fn s3_request_bucket_name(&self) -> &String {
        &self.s3_request_bucket_name
    }

    pub fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }

    pub fn sns_request_topic_arn(&self) -> &String {
        &self.sns_request_topic_arn
    }

    pub fn sqs_long_poll_wait_time(&self) -> usize {
        self.sqs_long_poll_wait_time
    }

    pub fn sqs_response_queue_url(&self) -> &String {
        &self.sqs_response_queue_url
    }

    pub async fn new(
        environment: String,
        s3_request_bucket_name: String,
        sns_request_topic_arn: String,
        sqs_long_poll_wait_time: usize,
        sqs_response_queue_url: String,
    ) -> Self {
        Self {
            environment: environment.to_owned(),
            s3_request_bucket_name,
            sdk: get_sdk_config().await,
            sns_request_topic_arn,
            sqs_long_poll_wait_time,
            sqs_response_queue_url,
        }
    }
}

impl Clone for AwsClientConfig {
    fn clone(&self) -> Self {
        Self {
            environment: self.environment.clone(),
            s3_request_bucket_name: self.s3_request_bucket_name.clone(),
            sns_request_topic_arn: self.sns_request_topic_arn.clone(),
            sqs_response_queue_url: self.sqs_response_queue_url.clone(),
            sdk: self.sdk.clone(),
            sqs_long_poll_wait_time: self.sqs_long_poll_wait_time,
        }
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

impl From<&AwsClientConfig> for S3Client {
    fn from(config: &AwsClientConfig) -> Self {
        let force_path_style =
            config.environment() != ENV_PROD && config.environment() != ENV_STAGE;
        let config_builder = S3ConfigBuilder::from(config.sdk())
            .retry_config(RetryConfig::standard().with_max_attempts(5))
            .force_path_style(force_path_style);

        S3Client::from_conf(config_builder.build())
    }
}

impl From<&AwsClientConfig> for SQSClient {
    fn from(config: &AwsClientConfig) -> Self {
        SQSClient::from_conf(
            Builder::from(config.sdk())
                .timeout_config(
                    TimeoutConfig::builder()
                        .operation_attempt_timeout(Duration::from_secs(
                            (config.sqs_long_poll_wait_time + 2) as u64,
                        ))
                        .build(),
                )
                .build(),
        )
    }
}

/// Returns AWS SDK configuration from a node configuration instance.
async fn get_sdk_config() -> aws_config::SdkConfig {
    let retry_config = RetryConfig::standard().with_max_attempts(20);
    aws_config::from_env()
        .retry_config(retry_config)
        .load()
        .await
}

#[cfg(test)]
mod tests {
    use super::AwsClientConfig;
    use crate::constants;

    async fn create_config() -> AwsClientConfig {
        AwsClientConfig::new(
            constants::DEFAULT_ENV.to_string(),
            constants::AWS_S3_REQUEST_BUCKET_NAME.to_string(),
            constants::AWS_SNS_REQUEST_TOPIC_ARN.to_string(),
            constants::AWS_SQS_LONG_POLL_WAIT_TIME,
            constants::AWS_SQS_RESPONSE_QUEUE_URL.to_string(),
        )
        .await
    }

    #[tokio::test]
    async fn test_config_new() {
        let _ = create_config().await;
    }
}
