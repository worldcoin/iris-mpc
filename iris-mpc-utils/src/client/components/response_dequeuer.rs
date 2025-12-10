use async_trait::async_trait;
use serde_json;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, ProcessRequestBatch, Request, RequestBatch, ResponseMessageBody,
};
use crate::aws::{types::SqsMessageInfo, AwsClient};

/// A component responsible for dequeuing system responses from network egress queues.
#[derive(Debug)]
pub struct ResponseDequeuer {
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
impl ProcessRequestBatch for ResponseDequeuer {
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ClientError> {
        let enqueued: Vec<&Request> = batch
            .requests_mut()
            .iter()
            .filter(|r| r.is_enqueued())
            .collect();
        tracing::info!("Enqueued count: {:?}", enqueued);

        let mut dequeued: usize = 0;
        while dequeued < enqueued.len() {
            while let Ok(sqs_msg) = self.aws_client.sqs_receive_messages(Some(1)).await {
                tracing::info!("TODO: correlate: {:?}", sqs_msg);
                dequeued += 1;
            }
            break;
        }

        Ok(())
    }
}

impl From<SqsMessageInfo> for ResponseMessageBody {
    fn from(msg: SqsMessageInfo) -> Self {
        match msg.kind().as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                ResponseMessageBody::IdentityDeletion(serde_json::from_str(&msg.body()).unwrap())
            }
            REAUTH_MESSAGE_TYPE => {
                ResponseMessageBody::Reauthorization(serde_json::from_str(&msg.body()).unwrap())
            }
            RESET_CHECK_MESSAGE_TYPE => {
                ResponseMessageBody::ResetCheck(serde_json::from_str(&msg.body()).unwrap())
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                ResponseMessageBody::ResetUpdate(serde_json::from_str(&msg.body()).unwrap())
            }
            UNIQUENESS_MESSAGE_TYPE => {
                ResponseMessageBody::Uniqueness(serde_json::from_str(&msg.body()).unwrap())
            }
            _ => panic!("Unsupported system response type"),
        }
    }
}
