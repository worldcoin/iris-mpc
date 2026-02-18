use async_trait::async_trait;
use serde_json;

use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use super::super::typeset::{
    Initialize, ProcessRequestBatch, Request, RequestBatch, ResponsePayload, ServiceClientError,
};
use crate::{
    aws::{types::SqsMessageInfo, AwsClient},
    constants::N_PARTIES,
};

/// Dequeues system responses from network egress queues.
pub(crate) struct ResponseDequeuer {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

#[async_trait]
impl Initialize for ResponseDequeuer {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        self.aws_client
            .sqs_purge_response_queue()
            .await
            .map_err(ServiceClientError::AwsServiceError)
    }
}

#[async_trait]
impl ProcessRequestBatch for ResponseDequeuer {
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ServiceClientError> {
        while batch.has_enqueued_items() {
            for sqs_msg in self
                .aws_client
                .sqs_receive_messages(Some(N_PARTIES))
                .await?
            {
                let response = ResponsePayload::from(&sqs_msg);

                // Validate response — allow errors to propagate upwards.
                response.validate()?;

                if let Some(idx) = batch.get_idx_of_correlated(&response) {
                    let is_now_complete = batch.requests_mut()[idx].record_response(&response);

                    if is_now_complete {
                        // If a Uniqueness request is now complete, activate any intra-batch
                        // pending children that were waiting for this parent.
                        if let Some((parent_key, serial_id)) =
                            extract_uniqueness_activation_info(batch, idx, &response)
                        {
                            batch.activate_pending(&parent_key, serial_id);
                        }
                    }

                    self.aws_client
                        .sqs_purge_response_queue_message(&sqs_msg)
                        .await?;
                } else {
                    tracing::warn!(
                        "Orphan response: no matching request found: {:#?}",
                        &response
                    );
                }
            }
        }

        Ok(())
    }
}

impl ResponseDequeuer {
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    /// Receives deletion responses from SQS during cleanup phase.
    /// Returns the serial IDs of confirmed deletions and purges the corresponding messages.
    pub(crate) async fn receive_deletion_responses(
        &self,
    ) -> Result<Vec<IrisSerialId>, ServiceClientError> {
        let mut confirmed = Vec::new();
        for sqs_msg in self
            .aws_client
            .sqs_receive_messages(Some(N_PARTIES))
            .await
            .map_err(ServiceClientError::AwsServiceError)?
        {
            let response = ResponsePayload::from(&sqs_msg);
            if let ResponsePayload::IdentityDeletion(result) = &response {
                if result.success {
                    confirmed.push(result.serial_id);
                }
            }
            self.aws_client
                .sqs_purge_response_queue_message(&sqs_msg)
                .await
                .map_err(ServiceClientError::AwsServiceError)?;
        }
        Ok(confirmed)
    }
}

/// Extracts the (parent_key, serial_id) needed to activate pending children when a Uniqueness
/// request completes. The parent_key is the request's label (Complex mode) or signup_id string
/// (Simple intra-batch mode).
fn extract_uniqueness_activation_info(
    batch: &RequestBatch,
    idx: usize,
    response: &ResponsePayload,
) -> Option<(String, IrisSerialId)> {
    if let Request::Uniqueness { signup_id, info, .. } = &batch.requests()[idx] {
        let serial_id = if let ResponsePayload::Uniqueness(result) = response {
            result.serial_id.or_else(|| {
                result
                    .matched_serial_ids
                    .as_ref()
                    .and_then(|m| m.first().copied())
            })?
        } else {
            return None;
        };

        // Use label for Complex mode, signup_id string for Simple intra-batch mode.
        let parent_key = info
            .label()
            .clone()
            .unwrap_or_else(|| signup_id.to_string());

        Some((parent_key, serial_id))
    } else {
        None
    }
}

impl From<&SqsMessageInfo> for ResponsePayload {
    fn from(msg: &SqsMessageInfo) -> Self {
        let body = msg.body();
        let kind = msg.kind();

        macro_rules! parse_response {
            ($variant:ident) => {
                ResponsePayload::$variant(serde_json::from_str(body).unwrap())
            };
        }

        match kind {
            IDENTITY_DELETION_MESSAGE_TYPE => parse_response!(IdentityDeletion),
            REAUTH_MESSAGE_TYPE => parse_response!(Reauthorization),
            RESET_CHECK_MESSAGE_TYPE => parse_response!(ResetCheck),
            RESET_UPDATE_MESSAGE_TYPE => parse_response!(ResetUpdate),
            UNIQUENESS_MESSAGE_TYPE => parse_response!(Uniqueness),
            _ => panic!("Unsupported system response type: {kind}"),
        }
    }
}
