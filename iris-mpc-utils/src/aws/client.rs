use std::collections::HashSet;

use async_from::AsyncFrom;
use async_stream::stream;
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
use crate::{client::AwsOptions, types::PublicKeyset};

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
    /// Returns the indexes of messages which were successfully published
    pub async fn sns_publish_json_batch(&self, messages: &[SnsMessageInfo]) -> Vec<usize> {
        use aws_sdk_sns::types::PublishBatchRequestEntry;
        const MAX_BATCH_SIZE: usize = 10;
        let mut indices = vec![];

        if messages.is_empty() {
            return indices;
        }

        // Process messages in chunks of up to 10
        for (chunk_idx, chunk) in messages.chunks(MAX_BATCH_SIZE).enumerate() {
            let entries: Result<Vec<_>, _> = chunk
                .iter()
                .enumerate()
                .map(|(idx, msg)| {
                    PublishBatchRequestEntry::builder()
                        .id(format!("msg-{}-{}", chunk_idx, idx))
                        .message(msg.body())
                        .message_group_id(msg.group_id())
                        .set_message_attributes(Some(create_sns_message_attributes(msg.kind())))
                        .build()
                })
                .collect();

            let entries = match entries {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("failed to build PublishBatchRequestEntry: {:?}", e);
                    return indices;
                }
            };

            let response = self
                .sns
                .publish_batch()
                .topic_arn(self.config().sns_request_topic_arn())
                .set_publish_batch_request_entries(Some(entries))
                .send()
                .await;

            let response = match response {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Failed to publish batch: {:?}", e);
                    return indices;
                }
            };

            // Map successful response IDs to original message indices
            if let Some(successful) = response.successful {
                for success in successful {
                    // Parse the ID format "msg-{chunk_idx}-{idx}" to get the original index
                    if let Some(id) = success.id {
                        if let Some(idx_str) = id.strip_prefix(&format!("msg-{}-", chunk_idx)) {
                            if let Ok(local_idx) = idx_str.parse::<usize>() {
                                let original_idx = chunk_idx * MAX_BATCH_SIZE + local_idx;
                                indices.push(original_idx);
                            }
                        }
                    }
                }
            }
        }

        indices
    }

    /// Purges all SQS response queues.
    pub async fn sqs_purge_response_queue(&self) -> Result<(), AwsClientError> {
        tracing::info!("AWS-SQS: purging system response queues");

        let purge_futures: Vec<_> = self
            .config()
            .sqs_response_queue_urls()
            .iter()
            .map(|queue_url| {
                let sqs = self.sqs.clone();
                let queue_url = queue_url.to_string();
                async move {
                    sqs.purge_queue()
                        .queue_url(&queue_url)
                        .send()
                        .await
                        .map(|_| ())
                        .map_err(|e| {
                            tracing::error!(
                                "AWS-SQS: response queue purge error for {}: {}",
                                queue_url,
                                e
                            );
                            AwsClientError::SqsPurgeQueueError(e.to_string())
                        })
                }
            })
            .collect();

        // Return first error if any, otherwise Ok
        for result in futures::future::join_all(purge_futures).await {
            result?;
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

    /// Creates a stream for a single SQS queue.
    ///
    /// Uses long polling with a fixed max batch size to efficiently receive messages.
    /// Messages are deleted from the queue immediately upon successful receipt and parsing.
    /// Only successfully deleted messages are yielded from the stream.
    fn sqs_stream(
        sqs: SQSClient,
        queue_url: String,
        long_poll_secs: i32,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<SqsMessageInfo, AwsClientError>> + Send>> {
        const MAX_SQS_BATCH: i32 = 10;

        Box::pin(stream! {
            loop {
                let response = match sqs
                    .receive_message()
                    .queue_url(&queue_url)
                    .wait_time_seconds(long_poll_secs)
                    .max_number_of_messages(MAX_SQS_BATCH)
                    .send()
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("AWS-SQS receive error (queue: {}): {}", queue_url, e);
                        yield Err(AwsClientError::SqsReceiveMessageError(e.to_string()));
                        continue;
                    }
                };

                let messages = response.messages.unwrap_or_default();
                if messages.is_empty() {
                    continue;
                }

                // Parse messages and collect receipt handles for deletion
                let (all_receipt_handles, parsed_messages) = parse_sqs_messages(&messages);

                // Batch delete ALL messages (including unparseable ones)
                if !all_receipt_handles.is_empty() {
                    let failed_ids =
                        match delete_messages_batch(&sqs, &queue_url, &all_receipt_handles)
                            .await
                        {
                            Ok(failed) => failed,
                            Err(e) => {
                                tracing::error!(
                                    "Failed to batch delete messages from queue {}: {}",
                                    queue_url,
                                    e
                                );
                                yield Err(e);
                                continue;
                            }
                        };

                    // Only yield successfully parsed AND successfully deleted messages
                    for (idx, msg_kind, msg_body, msg_receipt_handle) in parsed_messages {
                        let msg_id = format!("msg-{}", idx);
                        if !failed_ids.contains(&msg_id) {
                            let msg_info = SqsMessageInfo::new(
                                msg_kind,
                                msg_body,
                                queue_url.clone(),
                                msg_receipt_handle,
                            );
                            yield Ok(msg_info);
                        }
                    }
                }
            }
        })
    }

    /// Returns a stream that merges responses from all SQS response queues.
    ///
    /// Creates an independent stream for each queue that continuously polls, purges,
    /// and yields messages. All queue streams are merged into a single output stream.
    /// Uses long polling with a fixed max batch size for efficient message retrieval.
    pub fn sqs_response_stream(
        &self,
        long_poll_secs: i32,
    ) -> impl Stream<Item = Result<SqsMessageInfo, AwsClientError>> + '_ {
        let queue_urls = self.config().sqs_response_queue_urls().to_vec();
        let sqs = self.sqs.clone();

        // Create a stream for each queue
        let queue_streams: Vec<_> = queue_urls
            .into_iter()
            .map(|queue_url| Self::sqs_stream(sqs.clone(), queue_url, long_poll_secs))
            .collect();

        // Merge all queue streams into one
        futures::stream::select_all(queue_streams)
    }
}

/// Parses SQS messages and collects receipt handles for deletion.
/// Returns (all_receipt_handles, parsed_messages).
/// ALL messages have their receipt handles collected to prevent poison message redelivery.
#[allow(clippy::type_complexity)]
fn parse_sqs_messages(
    messages: &[aws_sdk_sqs::types::Message],
) -> (Vec<(usize, String)>, Vec<(usize, String, String, String)>) {
    let mut all_receipt_handles = Vec::new();
    let mut parsed_messages = Vec::new();

    for (idx, msg) in messages.iter().enumerate() {
        let receipt_handle = match msg.receipt_handle() {
            Some(h) => h.to_string(),
            None => {
                tracing::warn!("SQS message {} missing receipt_handle, cannot delete", idx);
                continue;
            }
        };

        // Always collect receipt handle for deletion
        all_receipt_handles.push((idx, receipt_handle.clone()));

        // Try to parse message
        let body_str = match msg.body() {
            Some(b) => b,
            None => {
                tracing::warn!(
                    "SQS message {} has no body, will delete without processing",
                    idx
                );
                continue;
            }
        };

        let decoded: serde_json::Value = match serde_json::from_str(body_str) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "SQS message {} has invalid JSON body: {}, will delete without processing",
                    idx,
                    e
                );
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
                tracing::warn!(
                    "SQS message {} missing message_type attribute, will delete without processing",
                    idx
                );
                continue;
            }
        };

        let msg_body = match decoded.get("Message").and_then(|v| v.as_str()) {
            Some(b) => b.to_string(),
            None => {
                tracing::warn!(
                    "SQS message {} missing Message field, will delete without processing",
                    idx
                );
                continue;
            }
        };

        // Successfully parsed - add to parsed messages
        parsed_messages.push((idx, msg_kind, msg_body, receipt_handle));
    }

    (all_receipt_handles, parsed_messages)
}

/// Batch deletes messages from SQS queue.
/// Returns the set of FAILED message IDs (messages that could not be deleted).
async fn delete_messages_batch(
    sqs: &SQSClient,
    queue_url: &str,
    all_receipt_handles: &[(usize, String)],
) -> Result<HashSet<String>, AwsClientError> {
    use aws_sdk_sqs::types::DeleteMessageBatchRequestEntry;

    let to_delete: Result<Vec<DeleteMessageBatchRequestEntry>, AwsClientError> =
        all_receipt_handles
            .iter()
            .map(|(idx, receipt_handle)| {
                DeleteMessageBatchRequestEntry::builder()
                    .id(format!("msg-{}", idx))
                    .receipt_handle(receipt_handle)
                    .build()
                    .map_err(|e| {
                        AwsClientError::SqsDeleteMessageError(format!(
                            "Failed to build DeleteMessageBatchRequestEntry: {}",
                            e
                        ))
                    })
            })
            .collect();

    let to_delete = to_delete?;

    let delete_response = sqs
        .delete_message_batch()
        .queue_url(queue_url)
        .set_entries(Some(to_delete))
        .send()
        .await
        .map_err(|e| AwsClientError::SqsDeleteMessageError(e.to_string()))?;

    // Collect failed message IDs
    let failed_ids: HashSet<String> = delete_response
        .failed
        .into_iter()
        .map(|f| {
            tracing::error!(
                "Failed to delete message {} from queue {}: {} - {}",
                f.id,
                queue_url,
                f.code,
                f.message.unwrap_or_default()
            );
            f.id
        })
        .collect();

    Ok(failed_ids)
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
