use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use serde::ser::Serialize;
use serde_json;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::{config::AwsClientConfig, errors::AwsClientError};

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
    pub(super) fn config(&self) -> &AwsClientConfig {
        &self.config
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

    /// Enqueues data to an S3 bucket.
    pub async fn s3_put_object(
        &self,
        s3_bucket: &str,
        s3_key: &str,
        s3_body: &[u8],
    ) -> Result<(), AwsClientError> {
        match self
            .s3
            .put_object()
            .bucket(s3_bucket)
            .key(s3_key)
            .body(ByteStream::new(SdkBody::from(s3_body)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("AWS-S3 upload error: {}", e);
                Err(AwsClientError::S3UploadError(
                    String::from(s3_key),
                    e.to_string(),
                ))
            }
        }
    }

    /// Enqueues a message upon an AWS SNS service topic.
    pub async fn sns_publish<T>(
        &self,
        message_type: &str,
        message_group_id: &str,
        message_body: T,
    ) -> Result<(), AwsClientError>
    where
        T: Sized + Serialize,
    {
        match self
            .sns
            .publish()
            .topic_arn(self.config().sns_request_topic_arn())
            .message_group_id(message_group_id)
            .message(serde_json::to_string(&message_body).unwrap())
            .set_message_attributes(Some(create_sns_message_attributes(message_type)))
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("AWS-SNS publish error: {}", e);
                Err(AwsClientError::SnsPublishError(e.to_string()))
            }
        }
    }

    /// Delete a message from an SQS queue.
    pub async fn sqs_delete_message(
        &self,
        sqs_receipt_handle: String,
    ) -> Result<(), AwsClientError> {
        match self
            .sqs
            .delete_message()
            .queue_url(self.config().sqs_response_queue_url())
            .receipt_handle(sqs_receipt_handle)
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("AWS-SQS delete message from queue error: {}", e);
                Err(AwsClientError::SqsDeleteMessageError(e.to_string()))
            }
        }
    }

    /// Dequeues a message from an SQS queue.
    pub async fn sqs_receive_message(&self) -> Result<(), AwsClientError> {
        match self
            .sqs
            .receive_message()
            .queue_url(self.config().sqs_response_queue_url())
            .wait_time_seconds(self.config().sqs_wait_time_seconds() as i32)
            .max_number_of_messages(1)
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("AWS-SQS receive message from queue error: {}", e);
                Err(AwsClientError::SqsReceiveMessageError(e.to_string()))
            }
        }
    }

    /// Purges an SQS queue.
    pub async fn sqs_purge_queue(&self) -> Result<(), AwsClientError> {
        match self
            .sqs
            .purge_queue()
            .queue_url(self.config().sqs_response_queue_url())
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("AWS-SQS queue purge error: {}", e);
                Err(AwsClientError::SqsPurgeQueueError(e.to_string()))
            }
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

#[cfg(test)]
mod tests {
    use super::super::{AwsClient, AwsClientConfig};

    impl AwsClient {
        fn assert_instance(&self) {
            assert!(self.s3.config().region().is_some());
            assert!(self.secrets_manager.config().region().is_some());
            assert!(self.sns.config().region().is_some());
            assert!(self.sqs.config().region().is_some());
        }

        async fn new_1() -> Self {
            Self::new(AwsClientConfig::new_1().await)
        }
    }

    #[tokio::test]
    async fn test_client_new() {
        AwsClient::new_1().await.assert_instance();
    }

    #[tokio::test]
    async fn test_client_new_and_clone() {
        AwsClient::new_1().await.clone().assert_instance();
    }
}
