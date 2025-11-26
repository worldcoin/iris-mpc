use std::time::Duration;

use aws_config::{retry::RetryConfig, timeout::TimeoutConfig, SdkConfig};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Builder, Client as SQSClient};

use iris_mpc_common::config::{ENV_PROD, ENV_STAGE};

/// Encpasulates AWS service client configuration.
#[derive(Clone, Debug)]
pub struct AwsClientConfig {
    /// Execution environment.
    environment: String,

    /// Base URL for downloading node encryption public keys.
    public_key_base_url: String,

    /// S3: request ingress queue URL.
    s3_request_bucket_name: String,

    /// SDK: associated AWS SDK configuration.
    sdk: SdkConfig,

    /// SNS: system request ingress queue topic.
    sns_request_topic_arn: String,

    /// SQS: long polling interval (seconds).
    sqs_long_poll_wait_time: usize,

    /// SQS: system response eqgress queue URL.
    sqs_response_queue_url: String,

    /// SQS: wait time (seconds) between receive message polling.
    sqs_wait_time_seconds: usize,
}

impl AwsClientConfig {
    pub(crate) fn environment(&self) -> &String {
        &self.environment
    }

    pub(crate) fn public_key_base_url(&self) -> &String {
        &self.public_key_base_url
    }

    pub(crate) fn s3_request_bucket_name(&self) -> &String {
        &self.s3_request_bucket_name
    }

    pub(super) fn sdk(&self) -> &SdkConfig {
        &self.sdk
    }

    pub(crate) fn sns_request_topic_arn(&self) -> &String {
        &self.sns_request_topic_arn
    }

    pub(crate) fn sqs_response_queue_url(&self) -> &String {
        &self.sqs_response_queue_url
    }

    pub(crate) fn sqs_wait_time_seconds(&self) -> usize {
        self.sqs_wait_time_seconds
    }

    pub async fn new(
        environment: String,
        public_key_base_url: String,
        s3_request_bucket_name: String,
        sns_request_topic_arn: String,
        sqs_long_poll_wait_time: usize,
        sqs_response_queue_url: String,
        sqs_wait_time_seconds: usize,
    ) -> Self {
        Self {
            environment,
            public_key_base_url,
            s3_request_bucket_name,
            sdk: get_sdk_config().await,
            sns_request_topic_arn,
            sqs_long_poll_wait_time,
            sqs_response_queue_url,
            sqs_wait_time_seconds,
        }
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

#[cfg(test)]
impl AwsClientConfig {
    #[cfg(test)]
    pub(crate) async fn new_1() -> Self {
        use crate::constants;

        let environment = String::from(constants::DEFAULT_ENV);
        let public_key_base_url = String::from(constants::AWS_PUBLIC_KEY_BASE_URL);
        let s3_request_bucket_name = String::from(constants::AWS_S3_REQUEST_BUCKET_NAME);
        let sns_request_topic_arn = String::from(constants::AWS_SNS_REQUEST_TOPIC_ARN);
        let sqs_long_poll_wait_time = constants::AWS_SQS_LONG_POLL_WAIT_TIME;
        let sqs_response_queue_url = String::from(constants::AWS_SQS_RESPONSE_QUEUE_URL);
        let sqs_wait_time_seconds = constants::AWS_SQS_LONG_POLL_WAIT_TIME;

        AwsClientConfig::new(
            environment,
            public_key_base_url,
            s3_request_bucket_name,
            sns_request_topic_arn,
            sqs_long_poll_wait_time,
            sqs_response_queue_url,
            sqs_wait_time_seconds,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::AwsClientConfig;

    #[tokio::test]
    async fn test_config_new() {
        let _ = AwsClientConfig::new_1().await;
    }

    #[tokio::test]
    async fn test_config_new_and_clone() {
        let _ = AwsClientConfig::new_1().await.clone();
    }
}
