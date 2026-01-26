use aws_config::retry::RetryConfig;
use aws_sdk_sns::config::Builder;
use aws_sdk_sns::Client as SNSClient;

pub fn create_sns_client(shared_config: &aws_config::SdkConfig, max_attempts: u32) -> SNSClient {
    tracing::info!("Creating SNS client with max attempts: {}", max_attempts);
    let sns_config = Builder::from(shared_config)
        .retry_config(RetryConfig::standard().with_max_attempts(max_attempts))
        .build();
    SNSClient::from_conf(sns_config)
}
