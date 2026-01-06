use async_trait::async_trait;
use serde_json;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    Initialize, ProcessRequestBatch, Request, RequestBatch, ResponsePayload, ServiceClientError,
    UniquenessReference,
};
use crate::{
    aws::{types::SqsMessageInfo, AwsClient},
    constants::N_PARTIES,
};

/// A component responsible for dequeuing system responses from network egress queues.
#[derive(Debug)]
pub(crate) struct ResponseDequeuer {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

impl ResponseDequeuer {
    /// Constructor.
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }
}

#[async_trait]
impl Initialize for ResponseDequeuer {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        self.aws_client
            .sqs_purge_queue()
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
                    )
                    .is_some()
                {
                    self.aws_client.sqs_purge_message(&sqs_msg).await?;
                }
            }
        }

        Ok(())
    }
}

impl ResponseDequeuer {
    /// Attempts to correlate an SQS response with a previously dispatched request.  If correlated
    /// then sets the corrleation and updates a child request (if found).
    fn maybe_correlate_response_and_update_child_request(
        &mut self,
        batch: &mut RequestBatch,
        response: ResponsePayload,
    ) -> Option<()> {
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
            Some(())
        } else {
            None
        }
    }

    fn maybe_update_child_request(&mut self, request: &mut Request, response: &ResponsePayload) {
        match request {
            Request::IdentityDeletion { uniqueness_ref, .. }
            | Request::Reauthorization { uniqueness_ref, .. }
            | Request::ResetUpdate { uniqueness_ref, .. } => {
                if let ResponsePayload::Uniqueness(result) = response {
                    let serial_id = result
                        .serial_id
                        .or_else(|| {
                            result
                                .matched_serial_ids
                                .as_ref()
                                .and_then(|matched| matched.first().copied())
                        })
                        .expect("Unmatched uniqueness request.");
                    *uniqueness_ref = UniquenessReference::IrisSerialId(serial_id);
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
            RESET_CHECK_MESSAGE_TYPE => parse_response!(ResetCheck),
            RESET_UPDATE_MESSAGE_TYPE => parse_response!(ResetUpdate),
            UNIQUENESS_MESSAGE_TYPE => parse_response!(Uniqueness),
            _ => panic!("Unsupported system response type: {kind}"),
        }
    }
}
