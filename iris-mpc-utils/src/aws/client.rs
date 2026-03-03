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

    /// Purges all SQS response queues.
    pub async fn sqs_purge_response_queue(&self) -> Result<(), AwsClientError> {
        tracing::debug!("AWS-SQS: purging system response queues");
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

    /// Dequeues messages from all SQS system response queues concurrently.
    ///
    /// Uses short polling (wait_time=0) so that empty queues return instantly
    /// rather than blocking the entire poll cycle. Sleeps briefly when no
    /// messages are found on any queue to avoid busy-looping.
    pub async fn sqs_receive_messages(
        &self,
        max_messages: Option<usize>,
        long_poll_secs: Option<i32>,
    ) -> Result<Vec<SqsMessageInfo>, AwsClientError> {
        let wait_time = long_poll_secs.unwrap_or(0);
        let futures: Vec<_> = self
            .config()
            .sqs_response_queue_urls()
            .iter()
            .map(|queue_url| {
                let sqs = self.sqs.clone();
                let queue_url = queue_url.clone();
                let max_msgs = max_messages.unwrap_or(1) as i32;
                async move {
                    let response = sqs
                        .receive_message()
                        .queue_url(&queue_url)
                        .wait_time_seconds(wait_time)
                        .max_number_of_messages(max_msgs)
                        .send()
                        .await
                        .map_err(|e| {
                            tracing::error!(
                                "AWS-SQS received message error (queue: {}) -> {}",
                                queue_url,
                                e
                            );
                            AwsClientError::SqsReceiveMessageError(e.to_string())
                        })?;
                    let messages: Vec<SqsMessageInfo> = response
                        .messages
                        .unwrap_or_default()
                        .into_iter()
                        .filter_map(|msg| {
                            let body_str = match msg.body() {
                                Some(b) => b,
                                None => {
                                    tracing::warn!("SQS message has no body, skipping");
                                    return None;
                                }
                            };
                            let decoded: serde_json::Value = match serde_json::from_str(body_str) {
                                Ok(v) => v,
                                Err(e) => {
                                    tracing::warn!("SQS message has invalid JSON body: {}", e);
                                    return None;
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
                                    return None;
                                }
                            };
                            let msg_body = match decoded.get("Message").and_then(|v| v.as_str()) {
                                Some(b) => b.to_string(),
                                None => {
                                    tracing::warn!("SQS message missing Message field");
                                    return None;
                                }
                            };
                            let msg_receipt_handle = match msg.receipt_handle() {
                                Some(h) => h.to_string(),
                                None => {
                                    tracing::warn!("SQS message missing receipt_handle");
                                    return None;
                                }
                            };
                            Some(SqsMessageInfo::new(
                                msg_kind,
                                msg_body,
                                queue_url.clone(),
                                msg_receipt_handle,
                            ))
                        })
                        .collect();
                    Ok::<_, AwsClientError>(messages)
                }
            })
            .collect();

        let results = futures::future::join_all(futures).await;
        let mut all_messages = Vec::new();
        for result in results {
            all_messages.extend(result?);
        }

        Ok(all_messages)
    }

    /// Creates a stream for a single SQS queue that continuously polls, purges, and yields messages.
    ///
    /// The stream will:
    /// 1. Poll the queue for messages using long polling
    /// 2. Parse and validate each message
    /// 3. Delete the message from the queue (purge)
    /// 4. Yield the parsed message info
    fn sqs_queue_stream(
        sqs: SQSClient,
        queue_url: String,
        max_messages: i32,
        long_poll_secs: i32,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<SqsMessageInfo, AwsClientError>> + Send>> {
        Box::pin(futures::stream::unfold(
            (sqs, queue_url, max_messages, long_poll_secs, Vec::new()),
            move |(sqs, queue_url, max_messages, long_poll_secs, mut buffer)| async move {
                loop {
                    // Yield buffered messages first
                    if let Some(msg) = buffer.pop() {
                        return Some((
                            Ok(msg),
                            (sqs, queue_url, max_messages, long_poll_secs, buffer),
                        ));
                    }

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
                                (sqs, queue_url, max_messages, long_poll_secs, buffer),
                            ));
                        }
                    };

                    let messages = response.messages.unwrap_or_default();
                    for msg in messages {
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

                        // Delete message from queue (purge) before yielding
                        if let Err(e) = sqs
                            .delete_message()
                            .queue_url(&queue_url)
                            .receipt_handle(&msg_receipt_handle)
                            .send()
                            .await
                        {
                            tracing::error!(
                                "Failed to delete message from queue {}: {}",
                                queue_url,
                                e
                            );
                            return Some((
                                Err(AwsClientError::SqsDeleteMessageError(e.to_string())),
                                (sqs, queue_url, max_messages, long_poll_secs, buffer),
                            ));
                        }

                        let msg_info = SqsMessageInfo::new(
                            msg_kind,
                            msg_body,
                            queue_url.clone(),
                            msg_receipt_handle,
                        );

                        buffer.push(msg_info);
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
    pub fn sqs_response_stream(
        &self,
        max_messages_per_queue: usize,
        long_poll_secs: i32,
    ) -> impl Stream<Item = Result<SqsMessageInfo, AwsClientError>> + '_ {
        let queue_urls = self.config().sqs_response_queue_urls().to_vec();
        let sqs = self.sqs.clone();
        let max_messages = max_messages_per_queue as i32;

        // Create a stream for each queue
        let queue_streams: Vec<_> = queue_urls
            .into_iter()
            .map(|queue_url| {
                Self::sqs_queue_stream(sqs.clone(), queue_url, max_messages, long_poll_secs)
            })
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
