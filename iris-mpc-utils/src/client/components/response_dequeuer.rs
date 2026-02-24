use async_trait::async_trait;
use serde_json;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    Initialize, ProcessRequestBatch, Request, RequestBatch, ResponsePayload, ServiceClientError,
    UniquenessRequestDescriptor,
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
                if self
                    .maybe_correlate_response_and_update_child_request(
                        batch,
                        ResponsePayload::from(&sqs_msg),
                    )?
                    .is_some()
                {
                    self.aws_client
                        .sqs_purge_response_queue_message(&sqs_msg)
                        .await?;
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

    /// Attempts to correlate an SQS response with a previously dispatched request.  If correlated
    /// then sets the correlation and updates a child request (if found).
    fn maybe_correlate_response_and_update_child_request(
        &mut self,
        batch: &mut RequestBatch,
        response: ResponsePayload,
    ) -> Result<Option<()>, ServiceClientError> {
        // Validate response ... allow errors to propogate upwards.
        response.validate()?;

        // If a correlation then update state accordingly. When fully correlated then
        // dispatch child request(s) if appropriate.
        if let Some(idx_of_correlated) = batch.get_idx_of_correlated(&response) {
            if batch.requests_mut()[idx_of_correlated]
                .set_correlation(&response)
                .is_some()
            {
                if let Some(idx_of_child) = batch.get_idx_of_child(idx_of_correlated) {
                    self.maybe_update_child_request(
                        &mut batch.requests_mut()[idx_of_child],
                        &response,
                    );
                }
            }
            Ok(Some(()))
        } else {
            tracing::warn!("Failed to correlate response: {:#?}", &response);
            Ok(None)
        }
    }

    /// Attempts to update a child request with data returned from it's parent's response.
    fn maybe_update_child_request(&mut self, request: &mut Request, response: &ResponsePayload) {
        match request {
            Request::IdentityDeletion { parent, .. }
            | Request::Reauthorization { parent, .. }
            | Request::ResetUpdate { parent, .. } => {
                if let ResponsePayload::Uniqueness(result) = response {
                    let serial_id = result
                        .serial_id
                        .or_else(|| {
                            result
                                .matched_serial_ids
                                .as_ref()
                                .and_then(|matched| matched.first().copied())
                        })
                        .unwrap_or_else(|| panic!("Unmatched uniqueness request: {:?}", result));
                    *parent = UniquenessRequestDescriptor::IrisSerialId(serial_id);
                }
            }
            _ => panic!("Unsupported parent data"),
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
            RESET_CHECK_MESSAGE_TYPE => parse_response!(IdentityMatchCheck),
            RESET_UPDATE_MESSAGE_TYPE => parse_response!(ResetUpdate),
            UNIQUENESS_MESSAGE_TYPE => parse_response!(Uniqueness),
            _ => panic!("Unsupported system response type: {kind}"),
        }
    }
}
