use serde::{Deserialize, Serialize};

use iris_mpc_common::IrisSerialId;

use super::RequestBatch;

/// Service client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClientConfiguration {
    // Associated AWS services configuration.
    aws: AwsConfiguration,

    // Associated request batch generation configuration.
    request_batch: RequestBatchConfiguration,
}

impl ServiceClientConfiguration {
    pub fn aws(&self) -> &AwsConfiguration {
        &self.aws
    }

    pub fn request_batch(&self) -> &RequestBatchConfiguration {
        &self.request_batch
    }
}

/// AWS specific configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AwsConfiguration {
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

impl AwsConfiguration {
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

/// Set of variants over inputs to request batch generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestBatchConfiguration {
    // Batches of single request type
    SimpleBatchKind {
        /// Number of request batches to generate.
        batch_count: usize,

        /// Determines type of requests to be included in each batch.
        batch_kind: String,

        /// Size of each batch.
        batch_size: usize,

        // A known serial identifier that allows response correlation to be bypassed.
        known_iris_serial_id: Option<IrisSerialId>,
    },
    /// A pre-built known set of request batches.
    KnownSet(Vec<RequestBatch>),
}
