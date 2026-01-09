use async_trait::async_trait;
use serde_json;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    Initialize, ProcessRequestBatch, RequestBatch, ResponsePayload, ServiceClientError,
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
                if batch
                    .correlate_and_update_child(ResponsePayload::from(&sqs_msg))
                    .is_some()
                {
                    self.aws_client.sqs_purge_message(&sqs_msg).await?;
                }
            }
        }

        Ok(())
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
