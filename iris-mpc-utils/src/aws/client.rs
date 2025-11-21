use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use serde_json;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::{
    config::AwsClientConfig,
    errors::AwsClientError,
    keys::download_public_keyset,
    types::{S3ObjectInfo, SnsMessageInfo},
};
use crate::types::PublicKeyset;

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Debug)]
pub struct AwsClient {
    /// Associated configuration.
    config: AwsClientConfig,

    /// Encryption public key set ... one per MPC node.
    public_keyset: Option<PublicKeyset>,

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
    pub(crate) fn config(&self) -> &AwsClientConfig {
        &self.config
    }

    /// Resturns set of MPC party public keys.
    pub(crate) fn public_keyset(&self) -> PublicKeyset {
        match self.public_keyset {
            Some(keys) => keys,
            _ => unreachable!(
                "Encryption public keys must be downloaded.  Use set_public_keyset function."
            ),
        }
    }

    pub fn new(config: AwsClientConfig) -> Self {
        Self {
            config: config.to_owned(),
            public_keyset: None,
            s3: S3Client::from(&config),
            secrets_manager: SecretsManagerClient::from(&config),
            sqs: SQSClient::from(&config),
            sns: SNSClient::from(&config),
        }
    }
}

impl AwsClient {
    /// Enqueues data to an S3 bucket.
    pub async fn s3_put_object(&self, s3_obj_info: S3ObjectInfo) -> Result<(), AwsClientError> {
        self.s3
            .put_object()
            .bucket(s3_obj_info.bucket())
            .key(s3_obj_info.key())
            .body(ByteStream::new(SdkBody::from(
                s3_obj_info.body().as_slice(),
            )))
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                tracing::error!("AWS-S3 upload error: {}", e);
                AwsClientError::S3UploadError(s3_obj_info.key().clone(), e.to_string())
            })
    }

    /// Downloads & assigns encryption keys.
    pub(crate) async fn set_public_keyset(&mut self) -> Result<(), AwsClientError> {
        let keys = download_public_keyset(self.config.public_key_base_url())
            .await
            .map_err(|e| {
                tracing::error!("MPC network public encryption keys download error: {}", e);
                AwsClientError::PublicKeysetDownloadError(e.to_string())
            })?;

        self.public_keyset = Some(keys);
        tracing::info!("MPC network public encryption keys downloaded");

        Ok(())
    }

    /// Enqueues a message upon an AWS SNS service topic.  The message body is JSON encodeable.
    pub async fn sns_publish_json(
        &self,
        sns_msg_info: SnsMessageInfo,
    ) -> Result<(), AwsClientError> {
        self.sns
            .publish()
            .topic_arn(self.config().sns_request_topic_arn())
            .message_group_id(sns_msg_info.group_id())
            .message(serde_json::to_string(sns_msg_info.body()).unwrap())
            .set_message_attributes(Some(create_sns_message_attributes(sns_msg_info.kind())))
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                tracing::error!("AWS-SNS publishing error: {}", e);
                AwsClientError::SnsPublishError(e.to_string())
            })
    }

    /// Delete a message from an SQS queue.
    pub async fn sqs_delete_message(&self, sqs_receipt_handle: &str) -> Result<(), AwsClientError> {
        let queue_url = self.config().sqs_response_queue_url();

        self.sqs
            .delete_message()
            .queue_url(queue_url)
            .receipt_handle(sqs_receipt_handle)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                tracing::error!("AWS-SQS delete message from queue error: {}", e);
                AwsClientError::SqsDeleteMessageError(e.to_string())
            })
    }

    /// Purges an SQS queue.
    pub async fn sqs_purge_response_queue(&self) -> Result<(), AwsClientError> {
        let queue_url = self.config().sqs_response_queue_url();

        self.sqs
            .purge_queue()
            .queue_url(queue_url)
            .send()
            .await
            .map(|_| {
                tracing::info!("AWS-SQS response queue purged: {}", queue_url);
            })
            .map_err(|e| {
                tracing::error!("AWS-SQS queue purge error: {}", e);
                AwsClientError::SqsPurgeQueueError(e.to_string())
            })
    }

    /// Dequeues a message from an SQS queue.
    pub async fn sqs_receive_message(&self) -> Result<(), AwsClientError> {
        self.sqs
            .receive_message()
            .queue_url(self.config().sqs_response_queue_url())
            .wait_time_seconds(self.config().sqs_wait_time_seconds() as i32)
            .max_number_of_messages(1)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                tracing::error!("AWS-SQS receive message from queue error: {}", e);
                AwsClientError::SqsReceiveMessageError(e.to_string())
            })
    }
}

impl Clone for AwsClient {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            public_keyset: None,
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
