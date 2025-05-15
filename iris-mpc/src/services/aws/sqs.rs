use aws_config::timeout::TimeoutConfig;
use aws_sdk_sqs::config::Builder;
use aws_sdk_sqs::Client as SQSClient;
use std::time::Duration;

pub fn create_sqs_client(
    shared_config: &aws_config::SdkConfig,
    wait_time_seconds: usize,
) -> SQSClient {
    // Creates an SQS client with a client-side operation attempt timeout. Per default, there are two retry
    // attempts, meaning that every operation has three tries in total. This configuration prevents the sqs
    // client from `await`ing forever on broken streams. (see <https://github.com/awslabs/aws-sdk-rust/issues/1094>)
    SQSClient::from_conf(
        Builder::from(shared_config)
            .timeout_config(
                TimeoutConfig::builder()
                    .operation_attempt_timeout(Duration::from_secs((wait_time_seconds + 2) as u64))
                    .build(),
            )
            .build(),
    )
}
