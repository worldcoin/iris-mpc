use serde::{Deserialize, Serialize};

/// AWS specific configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsOptions {
    /// Execution environment.
    environment: String,

    /// Base URL for downloading node encryption public keys.
    public_key_base_url: String,

    /// S3: request ingress queue URL.
    s3_request_bucket_name: String,

    /// SNS: system request ingress queue topic.
    sns_request_topic_arn: String,

    /// SQS: long polling interval (seconds).
    sqs_long_poll_wait_time: usize,

    /// SQS: system response eqgress queue URL.
    sqs_response_queue_url: String,

    /// SQS: wait time (seconds) between receive message polling.
    sqs_wait_time_seconds: usize,
}

impl AwsOptions {
    pub fn environment(&self) -> &String {
        &self.environment
    }

    pub fn public_key_base_url(&self) -> &String {
        &self.public_key_base_url
    }

    pub fn s3_request_bucket_name(&self) -> &String {
        &self.s3_request_bucket_name
    }

    pub fn sns_request_topic_arn(&self) -> &String {
        &self.sns_request_topic_arn
    }

    pub fn sqs_long_poll_wait_time(&self) -> &usize {
        &self.sqs_long_poll_wait_time
    }

    pub fn sqs_response_queue_url(&self) -> &String {
        &self.sqs_response_queue_url
    }

    pub fn sqs_wait_time_seconds(&self) -> &usize {
        &self.sqs_wait_time_seconds
    }
}
