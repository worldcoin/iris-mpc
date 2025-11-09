use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use serde::ser::Serialize;
use serde_json;
use thiserror::Error;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::config::AwsClientConfig;
use crate::misc::{log_error, log_info};

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct AwsClient {
    /// Associated configuration.
    config: AwsClientConfig,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    secrets_manager: SecretsManagerClient,

    /// Client for Amazon Simple Notification Service.
    sns: SNSClient,

    /// Client for Amazon Simple Queue Service.
    sqs: SQSClient,
}

impl AwsClient {
    pub fn config(&self) -> &AwsClientConfig {
        &self.config
    }

    pub fn s3(&self) -> &S3Client {
        &self.s3
    }

    pub fn secrets_manager(&self) -> &SecretsManagerClient {
        &self.secrets_manager
    }

    pub fn sns(&self) -> &SNSClient {
        &self.sns
    }

    pub fn sqs(&self) -> &SQSClient {
        &self.sqs
    }

    pub fn new(config: AwsClientConfig) -> Self {
        Self {
            config: config.to_owned(),
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }

    pub(super) fn log_error(&self, msg: &str) {
        log_error("Utils-AWS", msg);
    }

    pub(super) fn log_info(&self, msg: &str) {
        log_info("Utils-AWS", msg);
    }

    /// Enqueues a message upon an AWS SNS service topic.
    pub async fn sns_publish<T>(
        &self,
        message_type: &str,
        message_group_id: &str,
        message_payload: T,
    ) -> Result<(), AwsClientError>
    where
        T: Sized + Serialize,
    {
        match self
            .sns()
            .clone()
            .publish()
            .topic_arn(self.config().sns_request_topic_arn())
            .message_group_id(message_group_id)
            .message(serde_json::to_string(&message_payload).unwrap())
            .set_message_attributes(Some(create_sns_message_attributes(message_type)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => Err(AwsClientError::SnsPublishError {
                error: e.to_string(),
            }),
        }
    }

    /// Purges an SQS queue.
    pub async fn sqs_purge_queue(&self, queue_url: &String) -> Result<(), AwsClientError> {
        match self.sqs().purge_queue().queue_url(queue_url).send().await {
            Ok(_) => Ok(()),
            Err(e) => Err(AwsClientError::SqsPurgeQueueError {
                error: e.to_string(),
            }),
        }
    }

    /// Enqueues data to an S3 bucket.
    pub async fn s3_upload(
        &self,
        s3_bucket: &str,
        s3_key: &str,
        payload: &[u8],
    ) -> Result<(), AwsClientError> {
        match self
            .s3()
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(ByteStream::new(SdkBody::from(payload)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => Err(AwsClientError::S3UploadError {
                key: s3_key.to_string(),
                error: e.to_string(),
            }),
        }
    }
}

impl Clone for AwsClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            sqs: self.sqs.clone(),
            sns: self.sns.clone(),
            s3: self.s3.clone(),
            secrets_manager: self.secrets_manager.clone(),
        }
    }
}

#[derive(Error, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum AwsClientError {
    #[error("AWS SQS purge queue error: {}", .error)]
    SqsPurgeQueueError { error: String },

    #[error("AWS SNS publish error")]
    SnsPublishError { error: String },

    #[error("AWS S3 upload error: key={}: error={}", .key, .error)]
    S3UploadError { key: String, error: String },
}

#[cfg(test)]
mod tests {
    use super::super::{AwsClient, AwsClientConfig};
    use crate::constants::{self};

    fn assert_clients(clients: &AwsClient) {
        assert!(clients.s3().config().region().is_some());
        assert!(clients.secrets_manager().config().region().is_some());
        assert!(clients.sns().config().region().is_some());
        assert!(clients.sqs().config().region().is_some());
    }

    async fn create_client() -> AwsClient {
        AwsClient::new(create_config().await)
    }

    async fn create_config() -> AwsClientConfig {
        AwsClientConfig::new(
            constants::DEFAULT_ENV.to_string(),
            constants::AWS_S3_REQUEST_BUCKET_NAME.to_string(),
            constants::AWS_SQS_RESPONSE_QUEUE_URL.to_string(),
            constants::AWS_SQS_LONG_POLL_WAIT_TIME,
            constants::AWS_SNS_REQUEST_TOPIC_ARN.to_string(),
        )
        .await
    }

    #[tokio::test]
    async fn test_client_new() {
        let clients = create_client().await;
        assert_clients(&clients);
    }

    #[tokio::test]
    async fn test_client_clone() {
        let clients = create_client().await.clone();
        assert_clients(&clients);
    }
}
