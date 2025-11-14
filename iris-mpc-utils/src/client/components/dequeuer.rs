use super::super::{errors::ServiceClientError, types::RequestBatch};
use crate::aws::AwsClient;

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

    /// Dequeues responses from system egress queues.
    pub async fn dequeue(&self, _batch: &RequestBatch) -> Result<(), ServiceClientError> {
        let msg = self.aws_client.sqs_receive_message().await;
        println!("AWS-SQS receive message event: {:?}", msg);

        Ok(())
    }
}
