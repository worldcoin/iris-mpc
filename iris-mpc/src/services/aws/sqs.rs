use aws_config::timeout::TimeoutConfig;
use aws_sdk_sqs::config::{Builder, Region};
use aws_sdk_sqs::Client as SQSClient;
use std::time::Duration;

pub fn create_sqs_client(
    shared_config: &aws_config::SdkConfig,
    wait_time_seconds: usize,
    region_override: Option<Region>,
) -> SQSClient {
    let mut builder = Builder::from(shared_config);
    if let Some(region) = region_override {
        builder = builder.region(region);
    }
    // Creates an SQS client with a client-side operation attempt timeout. Per default, there are two retry
    // attempts, meaning that every operation has three tries in total. This configuration prevents the sqs
    // client from `await`ing forever on broken streams. (see <https://github.com/awslabs/aws-sdk-rust/issues/1094>)
    SQSClient::from_conf(
        builder
            .timeout_config(
                TimeoutConfig::builder()
                    .operation_attempt_timeout(Duration::from_secs((wait_time_seconds + 2) as u64))
                    .build(),
            )
            .build(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_region_override_preserves_shared_config_and_timeout() {
        let shared_config = aws_config::SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .region(Region::new("eu-north-1"))
            .build();

        for (region_override, expected_region) in [
            (None, "eu-north-1"),
            (Some(Region::new("eu-central-1")), "eu-central-1"),
        ] {
            let client = create_sqs_client(&shared_config, 20, region_override);
            assert_eq!(client.config().region().unwrap().as_ref(), expected_region);
            assert_eq!(
                client
                    .config()
                    .timeout_config()
                    .unwrap()
                    .operation_attempt_timeout(),
                Some(Duration::from_secs(22))
            );
            assert_eq!(shared_config.region().unwrap().as_ref(), "eu-north-1");
        }
    }
}
