use std::cmp;

use async_from::AsyncFrom;
use aws_sdk_s3::{
    primitives::{ByteStream, SdkBody},
    Client as S3Client,
};
use aws_sdk_secretsmanager::Client as SecretsManagerClient;
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::Client as SQSClient;
use futures::stream::Stream;
use serde_json;

use iris_mpc_common::helpers::smpc_response::create_sns_message_attributes;

use super::{
    config::AwsClientConfig,
    errors::AwsClientError,
    keys::download_public_keyset,
    types::{S3ObjectInfo, SnsMessageInfo, SqsMessageInfo},
};
use crate::{client::AwsOptions, constants, types::PublicKeyset};

/// Encapsulates access to a node's set of AWS service clients.
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

#[async_from::async_trait]
impl AsyncFrom<AwsOptions> for AwsClient {
    async fn async_from(opts: AwsOptions) -> Self {
        AwsClient::new(AwsClientConfig::async_from(opts).await)
    }
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
            .body(ByteStream::new(SdkBody::from(s3_obj_info.body())))
            .send()
            .await
            .map(|_| ())
            .map_err(|e| {
                tracing::error!("AWS-S3 upload error: {}", e);
                AwsClientError::S3UploadError(s3_obj_info.key().to_string(), e.to_string())
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
        self.sns
            .publish()
            .topic_arn(self.config().sns_request_topic_arn())
            .message_group_id(sns_msg_info.group_id())
            .message(sns_msg_info.body())
            .set_message_attributes(Some(create_sns_message_attributes(sns_msg_info.kind())))
            .send()
            .await
            .map(|_| {})
            .map_err(|e| AwsClientError::SnsPublishError(e.to_string()))
    }

    /// Enqueues multiple messages upon an AWS SNS service topic using batch publish.
    /// AWS SNS supports up to 10 messages per batch.
    /// Returns the number of successfully published messages.
    pub async fn sns_publish_json_batch(
        &self,
        messages: &[SnsMessageInfo],
    ) -> Result<usize, AwsClientError> {
        use aws_sdk_sns::types::PublishBatchRequestEntry;

        if messages.is_empty() {
            return Ok(0);
        }

        const MAX_BATCH_SIZE: usize = 10;
        let mut total_published = 0;

        // Process messages in chunks of up to 10
        for (chunk_idx, chunk) in messages.chunks(MAX_BATCH_SIZE).enumerate() {
            let entries: Vec<PublishBatchRequestEntry> = chunk
                .iter()
                .enumerate()
                .map(|(idx, msg)| {
                    PublishBatchRequestEntry::builder()
                        .id(format!("msg-{}-{}", chunk_idx, idx))
                        .message(msg.body())
                        .message_group_id(msg.group_id())
                        .set_message_attributes(Some(create_sns_message_attributes(msg.kind())))
                        .build()
                        .expect("Failed to build PublishBatchRequestEntry")
                })
                .collect();

            let response = self
                .sns
                .publish_batch()
                .topic_arn(self.config().sns_request_topic_arn())
                .set_publish_batch_request_entries(Some(entries))
                .send()
                .await
                .map_err(|e| AwsClientError::SnsPublishBatchError(e.to_string()))?;

            // Count successful publishes
            let successful = response.successful.unwrap_or_default().len();
            total_published += successful;

            // Log any failures
            if let Some(failed) = response.failed {
                for failure in failed {
                    tracing::error!(
                        "SNS batch publish failed for message {}: {} - {}",
                        failure.id,
                        failure.code,
                        failure.message.unwrap_or_default()
                    );
                }
            }

            // If any messages in this chunk failed, return error
            if successful < chunk.len() {
                return Err(AwsClientError::SnsPublishBatchError(format!(
                    "Only {}/{} messages published successfully in chunk {}",
                    successful,
                    chunk.len(),
                    chunk_idx
                )));
            }
        }

        Ok(total_published)
    }

    /// Purges all SQS response queues.
    pub async fn sqs_purge_response_queue(&self) -> Result<(), AwsClientError> {
        tracing::info!("AWS-SQS: purging system response queues");
        for queue_url in self.config().sqs_response_queue_urls() {
            // Check if queue has any messages before purging
            let attributes = self
                .sqs
                .get_queue_attributes()
                .queue_url(queue_url)
                .attribute_names(
                    aws_sdk_sqs::types::QueueAttributeName::ApproximateNumberOfMessages,
                )
                .send()
                .await
                .map_err(|e| {
                    tracing::error!("AWS-SQS: get queue attributes error: {}", e);
                    AwsClientError::SqsGetQueueAttributesError(e.to_string())
                })?;

            let message_count = attributes
                .attributes()
                .and_then(|attrs| {
                    attrs.get(&aws_sdk_sqs::types::QueueAttributeName::ApproximateNumberOfMessages)
                })
                .and_then(|count| count.parse::<i32>().ok())
                .unwrap_or(0);

            if message_count == 0 {
                tracing::debug!("AWS-SQS: queue is empty, skipping purge for {}", queue_url);
                continue;
            }

            self.sqs
                .purge_queue()
                .queue_url(queue_url)
                .send()
                .await
                .map(|_| {})
                .map_err(|e| {
                    tracing::error!("AWS-SQS: response queue purge error: {}", e);
                    AwsClientError::SqsPurgeQueueError(e.to_string())
                })?;
        }
        tracing::info!("AWS-SQS: purge finished");
        Ok(())
    }

    /// Purges SQS response queue message.
    pub async fn sqs_purge_response_queue_message(
        &self,
        sqs_msg: &SqsMessageInfo,
    ) -> Result<(), AwsClientError> {
        self.sqs
            .delete_message()
            .queue_url(sqs_msg.queue_url())
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

    /// Creates a stream for a single SQS queue with dynamic batch sizing.
    /// Checks queue depth before each receive and adjusts batch size accordingly.
    fn sqs_stream(
        sqs: SQSClient,
        queue_url: String,
        long_poll_secs: i32,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<SqsMessageInfo, AwsClientError>> + Send>> {
        const MAX_SQS_BATCH: i32 = 10;

        Box::pin(futures::stream::unfold(
            (sqs, queue_url, long_poll_secs, Vec::new()),
            move |(sqs, queue_url, long_poll_secs, mut buffer)| async move {
                loop {
                    // Yield buffered messages first
                    if let Some(msg) = buffer.pop() {
                        return Some((Ok(msg), (sqs, queue_url, long_poll_secs, buffer)));
                    }

                    // Check queue depth to calculate optimal batch size
                    let max_messages = match sqs
                        .get_queue_attributes()
                        .queue_url(&queue_url)
                        .attribute_names(
                            aws_sdk_sqs::types::QueueAttributeName::ApproximateNumberOfMessages,
                        )
                        .send()
                        .await
                    {
                        Ok(attributes) => {
                            let message_count = attributes
                                .attributes()
                                .and_then(|attrs| {
                                    attrs.get(&aws_sdk_sqs::types::QueueAttributeName::ApproximateNumberOfMessages)
                                })
                                .and_then(|count| count.parse::<i32>().ok())
                                .unwrap_or(0);

                            let message_count = cmp::min(MAX_BATCH_COUNT, message_count);
                            cmp::max(message_count, 1)
                        }
                        Err(e) => {
                            tracing::debug!(
                                "Failed to get queue attributes for {}: {}, using default",
                                queue_url,
                                e
                            );
                            1
                        }
                    };

                    // Poll the queue
                    let response = match sqs
                        .receive_message()
                        .queue_url(&queue_url)
                        .wait_time_seconds(long_poll_secs)
                        .max_number_of_messages(max_messages)
                        .send()
                        .await
                    {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::error!("AWS-SQS receive error (queue: {}): {}", queue_url, e);
                            return Some((
                                Err(AwsClientError::SqsReceiveMessageError(e.to_string())),
                                (sqs, queue_url, long_poll_secs, buffer),
                            ));
                        }
                    };

                    let messages = response.messages.unwrap_or_default();

                    // Parse all messages first
                    let mut parsed_messages = Vec::new();
                    for (idx, msg) in messages.iter().enumerate() {
                        // Parse message
                        let body_str = match msg.body() {
                            Some(b) => b,
                            None => {
                                tracing::warn!("SQS message has no body, skipping");
                                continue;
                            }
                        };

                        let decoded: serde_json::Value = match serde_json::from_str(body_str) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!("SQS message has invalid JSON body: {}", e);
                                continue;
                            }
                        };

                        let msg_kind = match decoded
                            .get("MessageAttributes")
                            .and_then(|v| v.get("message_type"))
                            .and_then(|v| v.get("Value"))
                            .and_then(|v| v.as_str())
                        {
                            Some(k) => k.to_string(),
                            None => {
                                tracing::warn!("SQS message missing message_type attribute");
                                continue;
                            }
                        };

                        let msg_body = match decoded.get("Message").and_then(|v| v.as_str()) {
                            Some(b) => b.to_string(),
                            None => {
                                tracing::warn!("SQS message missing Message field");
                                continue;
                            }
                        };

                        let msg_receipt_handle = match msg.receipt_handle() {
                            Some(h) => h.to_string(),
                            None => {
                                tracing::warn!("SQS message missing receipt_handle");
                                continue;
                            }
                        };

                        parsed_messages.push((idx, msg_kind, msg_body, msg_receipt_handle));
                    }

                    // Batch delete all messages
                    if !parsed_messages.is_empty() {
                        use aws_sdk_sqs::types::DeleteMessageBatchRequestEntry;

                        let delete_entries: Vec<DeleteMessageBatchRequestEntry> = parsed_messages
                            .iter()
                            .map(|(idx, _, _, receipt_handle)| {
                                DeleteMessageBatchRequestEntry::builder()
                                    .id(format!("msg-{}", idx))
                                    .receipt_handle(receipt_handle)
                                    .build()
                                    .expect("Failed to build DeleteMessageBatchRequestEntry")
                            })
                            .collect();

                        match sqs
                            .delete_message_batch()
                            .queue_url(&queue_url)
                            .set_entries(Some(delete_entries))
                            .send()
                            .await
                        {
                            Ok(delete_response) => {
                                // Log any failures
                                for failure in delete_response.failed {
                                    tracing::error!(
                                        "Failed to delete message {} from queue {}: {} - {}",
                                        failure.id,
                                        queue_url,
                                        failure.code,
                                        failure.message.unwrap_or_default()
                                    );
                                }

                                // Add successfully deleted messages to buffer
                                for (_, msg_kind, msg_body, msg_receipt_handle) in parsed_messages {
                                    let msg_info = SqsMessageInfo::new(
                                        msg_kind,
                                        msg_body,
                                        queue_url.clone(),
                                        msg_receipt_handle,
                                    );
                                    buffer.push(msg_info);
                                }
                            }
                            Err(e) => {
                                tracing::error!(
                                    "Failed to batch delete messages from queue {}: {}",
                                    queue_url,
                                    e
                                );
                                return Some((
                                    Err(AwsClientError::SqsDeleteMessageError(e.to_string())),
                                    (sqs, queue_url, long_poll_secs, buffer),
                                ));
                            }
                        }
                    }

                    // Continue loop to yield buffered messages
                    if !buffer.is_empty() {
                        continue;
                    }
                }
            },
        ))
    }

    /// Returns a stream that merges responses from all SQS response queues.
    ///
    /// Creates an independent stream for each queue that continuously polls, purges,
    /// and yields messages. All queue streams are merged into a single output stream.
    /// The batch size is dynamically calculated based on available messages.
    pub fn sqs_response_stream(
        &self,
        long_poll_secs: i32,
    ) -> impl Stream<Item = Result<SqsMessageInfo, AwsClientError>> + '_ {
        let queue_urls = self.config().sqs_response_queue_urls().to_vec();
        let sqs = self.sqs.clone();

        // Create a stream for each queue with dynamic batch sizing
        let queue_streams: Vec<_> = queue_urls
            .into_iter()
            .map(|queue_url| Self::sqs_stream(sqs.clone(), queue_url, long_poll_secs))
            .collect();

        // Merge all queue streams into one
        futures::stream::select_all(queue_streams)
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
