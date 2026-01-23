use aws_config::retry::RetryConfig;
use aws_sdk_sns::config::Builder;
use aws_sdk_sns::Client as SNSClient;

pub fn create_sns_client(shared_config: &aws_config::SdkConfig) -> SNSClient {
    let sns_config = Builder::from(shared_config)
        .retry_config(RetryConfig::standard().with_max_attempts(5))
        .build();
    SNSClient::from_conf(sns_config)
}
