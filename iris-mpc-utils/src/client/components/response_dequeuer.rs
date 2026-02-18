use async_trait::async_trait;
use serde_json;
use std::collections::{HashMap, HashSet};

use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use super::super::typeset::{
    Initialize, Request, RequestBatch, ResponsePayload, ServiceClientError,
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

impl ResponseDequeuer {
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    pub(crate) async fn process_batch(
        &mut self,
        batch: &mut RequestBatch,
        live_serial_ids: &mut HashSet<IrisSerialId>,
        label_resolutions: &mut HashMap<String, IrisSerialId>,
    ) -> Result<(), ServiceClientError> {
        while batch.has_enqueued_items() {
            for sqs_msg in self
                .aws_client
                .sqs_receive_messages(Some(N_PARTIES))
                .await?
            {
                let response = ResponsePayload::from(&sqs_msg);

                if let Some(idx) = batch.get_idx_of_correlated(&response) {
                    // Warn on error response but still record it — errors count as responses.
                    if response.is_error() {
                        tracing::warn!(
                            "{} :: Error response -> Node-{}: {}",
                            &batch.requests()[idx],
                            response.node_id(),
                            response.error_reason().unwrap_or("unknown error"),
                        );
                    }

                    let is_now_complete = batch.requests_mut()[idx].record_response(&response);

                    if is_now_complete {
                        let has_error = batch.requests()[idx].has_error_response();

                        // Check serial ID consistency across nodes for completed Uniqueness requests.
                        check_uniqueness_serial_id_consistency(batch, idx);

                        // Activate or drop intra-batch pending children based on outcome.
                        if let Some((parent_key, serial_id)) =
                            extract_uniqueness_activation_info(batch, idx)
                        {
                            if has_error {
                                tracing::warn!(
                                    "{} :: Completed with errors — dropping pending children",
                                    &batch.requests()[idx],
                                );
                                batch.drop_pending(&parent_key);
                            } else {
                                batch.activate_pending(&parent_key, serial_id);
                                live_serial_ids.insert(serial_id);

                                // Record label resolution for cross-batch parent lookups.
                                if let Request::Uniqueness { info, .. } = &batch.requests()[idx] {
                                    if let Some(label) = info.label() {
                                        label_resolutions.insert(label.clone(), serial_id);
                                    }
                                }
                            }
                        }

                        if let ResponsePayload::IdentityDeletion(result) = &response {
                            if result.success {
                                live_serial_ids.remove(&result.serial_id);
                            }
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
/// request completes. Uses all stored node responses to find a serial_id. Returns None if the
/// request is not a Uniqueness type or if no serial_id is present in any stored response.
/// The parent_key is the request's label (Complex mode) or signup_id string (Simple mode).
fn extract_uniqueness_activation_info(
    batch: &RequestBatch,
    idx: usize,
) -> Option<(String, IrisSerialId)> {
    if let Request::Uniqueness {
        signup_id, info, ..
    } = &batch.requests()[idx]
    {
        // Search all stored node responses for a serial_id.
        let serial_id = info.responses().iter().find_map(|opt| {
            if let Some(ResponsePayload::Uniqueness(result)) = opt {
                result
                    .serial_id
                    .or_else(|| result.matched_serial_ids.as_ref()?.first().copied())
            } else {
                None
            }
        })?;

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

/// Warns if the serial_ids returned by different nodes for a completed Uniqueness request differ.
fn check_uniqueness_serial_id_consistency(batch: &RequestBatch, idx: usize) {
    if let Request::Uniqueness { info, .. } = &batch.requests()[idx] {
        let serial_ids: Vec<IrisSerialId> = info
            .responses()
            .iter()
            .filter_map(|opt| {
                if let Some(ResponsePayload::Uniqueness(result)) = opt {
                    result.serial_id
                } else {
                    None
                }
            })
            .collect();

        if serial_ids.len() > 1 {
            let first = serial_ids[0];
            if serial_ids.iter().any(|&s| s != first) {
                tracing::warn!(
                    "{} :: Inconsistent serial_ids across nodes: {:?}",
                    info,
                    serial_ids,
                );
            }
        }
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
