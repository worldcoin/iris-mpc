use async_trait::async_trait;
use serde_json;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{ClientError, ProcessRequestBatch, RequestBatch, ResponseBody};
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
        while batch.is_enqueueable() {
            for sqs_msg in self.aws_client.sqs_receive_messages(Some(1)).await? {
                batch.maybe_set_response(ResponseBody::from(&sqs_msg));
            }
        }

        Ok(())
    }
}

impl From<&SqsMessageInfo> for ResponseBody {
    fn from(msg: &SqsMessageInfo) -> Self {
        match msg.kind().as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => {
                ResponseBody::IdentityDeletion(serde_json::from_str(&msg.body()).unwrap())
            }
            REAUTH_MESSAGE_TYPE => {
                ResponseBody::Reauthorization(serde_json::from_str(&msg.body()).unwrap())
            }
            RESET_CHECK_MESSAGE_TYPE => {
                ResponseBody::ResetCheck(serde_json::from_str(&msg.body()).unwrap())
            }
            RESET_UPDATE_MESSAGE_TYPE => {
                ResponseBody::ResetUpdate(serde_json::from_str(&msg.body()).unwrap())
            }
            UNIQUENESS_MESSAGE_TYPE => {
                ResponseBody::Uniqueness(serde_json::from_str(&msg.body()).unwrap())
            }
            _ => panic!("Unsupported system response type"),
        }
    }
}
