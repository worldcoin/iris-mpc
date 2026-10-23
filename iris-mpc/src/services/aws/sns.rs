use aws_config::retry::RetryConfig;
use aws_sdk_sns::config::{Builder, Region};
use aws_sdk_sns::Client as SNSClient;

pub fn create_sns_client(
    shared_config: &aws_config::SdkConfig,
    max_attempts: u32,
    region_override: Option<Region>,
) -> SNSClient {
    tracing::info!("Creating SNS client with max attempts: {}", max_attempts);
    let mut builder = Builder::from(shared_config);
    if let Some(region) = region_override {
        builder = builder.region(region);
    }
    let sns_config = builder
        .retry_config(RetryConfig::standard().with_max_attempts(max_attempts))
        .build();
    SNSClient::from_conf(sns_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::aws::s3::create_s3_client;

    #[test]
    fn result_region_override_preserves_shared_clients_endpoint_and_retries() {
        let shared_config = aws_config::SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .region(Region::new("eu-north-1"))
            .endpoint_url("http://localhost:4566")
            .build();

        for (region_override, expected_region) in [
            (None, "eu-north-1"),
            (Some(Region::new("eu-central-1")), "eu-central-1"),
        ] {
            let sns = create_sns_client(&shared_config, 7, region_override);
            assert_eq!(sns.config().region().unwrap().as_ref(), expected_region);
            assert_eq!(sns.config().retry_config().unwrap().max_attempts(), 7);
            assert_eq!(shared_config.endpoint_url(), Some("http://localhost:4566"));
            let s3 = create_s3_client(&shared_config, false, None);
            assert_eq!(s3.config().region().unwrap().as_ref(), "eu-north-1");
            assert_eq!(shared_config.region().unwrap().as_ref(), "eu-north-1");
        }
    }
}
