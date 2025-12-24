use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{types::Message as SqsMessage, Client as SQSClient};
use serde_json;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::{
    config::AwsClientConfig,
    errors::AwsClientError,
    keys::download_public_keyset,
    types::{S3ObjectInfo, SnsMessageInfo, SqsMessageInfo},
};
use crate::types::PublicKeyset;

/// Encpasulates access to a node's set of AWS service clients.
#[derive(Clone, Debug)]
pub struct AwsClient {
    /// Associated configuration.
    config: AwsClientConfig,

    /// Encryption public key set ... one per MPC node.
    public_keyset: Option<PublicKeyset>,

    /// Client for Amazon Simple Storage Service.
    s3: S3Client,

    /// Client for AWS Secrets Manager.
    #[allow(dead_code)]
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
    pub async fn s3_put_object(&self, s3_obj_info: &S3ObjectInfo) -> Result<(), AwsClientError> {
        tracing::debug!("AWS-S3: putting object -> {}", s3_obj_info);
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
        self.public_keyset = Some(
            download_public_keyset(self.config.public_key_base_url())
                .await
                .map_err(|e| {
                    tracing::error!("MPC public keys download error: {}", e);
                    AwsClientError::PublicKeysetDownloadError(e.to_string())
                })?,
        );

        Ok(())
    }

    /// Enqueues a message upon an AWS SNS service topic.  The message body is JSON encodeable.
    pub async fn sns_publish_json(
        &self,
        sns_msg_info: SnsMessageInfo,
    ) -> Result<(), AwsClientError> {
        tracing::debug!("AWS-SNS: publishing message -> {}", sns_msg_info);
        self.sns
            .publish()
            .topic_arn(self.config().sns_request_topic_arn())
            .message_group_id(sns_msg_info.group_id())
            .message(sns_msg_info.body())
            .set_message_attributes(Some(create_sns_message_attributes(sns_msg_info.kind())))
            .send()
            .await
            .map(|_| {})
            .map_err(|e| {
                tracing::error!("AWS-SNS publishing error: {}", e);
                AwsClientError::SnsPublishError(e.to_string())
            })
    }

    /// Purges a response message from an SQS queue.
    pub async fn sqs_purge_message(&self, sqs_msg: &SqsMessageInfo) -> Result<(), AwsClientError> {
        self.sqs
            .delete_message()
            .queue_url(self.config().sqs_response_queue_url())
            .receipt_handle(sqs_msg.receipt_handle())
            .send()
            .await
            .map(|_| {
                tracing::debug!("AWS-SQS: purged message -> {}", sqs_msg.kind());
            })
            .map_err(|e| {
                tracing::error!("AWS-SQS: purged message -> error: {}", e);
                AwsClientError::SqsDeleteMessageError(e.to_string())
            })
    }

    /// Purges an SQS queue.
    pub async fn sqs_purge_queue(&self) -> Result<(), AwsClientError> {
        tracing::debug!("AWS-SQS: purging response queue");
        self.sqs
            .purge_queue()
            .queue_url(self.config().sqs_response_queue_url())
            .send()
            .await
            .map(|_| {})
            .map_err(|e| {
                tracing::error!("AWS-SQS: response queue purge error: {}", e);
                AwsClientError::SqsPurgeQueueError(e.to_string())
            })
    }

    /// Dequeues messages from SQS system response queue.
    pub async fn sqs_receive_messages(
        &self,
        max_messages: Option<usize>,
    ) -> Result<impl Iterator<Item = SqsMessageInfo>, AwsClientError> {
        let response = self
            .sqs
            .receive_message()
            .queue_url(self.config().sqs_response_queue_url())
            .wait_time_seconds(self.config().sqs_wait_time_seconds() as i32)
            .max_number_of_messages(max_messages.unwrap_or(1) as i32)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("AWS-SQS received message error -> {}", e);
                AwsClientError::SqsReceiveMessageError(e.to_string())
            })?;
        let mapped = response
            .messages
            .unwrap_or_default()
            .into_iter()
            .map(|msg| {
                let msg = SqsMessageInfo::from(&msg);
                tracing::debug!("AWS-SQS: received message -> {}", msg);
                msg
            });

        Ok(mapped)
    }
}

impl From<&SqsMessage> for SqsMessageInfo {
    fn from(msg: &SqsMessage) -> Self {
        let decoded: serde_json::Value =
            serde_json::from_str(msg.body().expect("Empty JSON string"))
                .expect("Invalid JSON string");
        let msg_kind = decoded
            .get("MessageAttributes")
            .and_then(|v| v.get("message_type"))
            .and_then(|v| v.get("Value"))
            .and_then(|v| v.as_str())
            .expect("Missing message_type")
            .to_string();
        let msg_body = decoded
            .get("Message")
            .and_then(|v| v.as_str())
            .expect("Missing Message")
            .to_string();
        let msg_receipt_handle = msg
            .receipt_handle()
            .expect("Missing receipt_handle")
            .to_string();

        SqsMessageInfo::new(msg_kind, msg_body, msg_receipt_handle)
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

        pub(crate) async fn new_1() -> Self {
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
