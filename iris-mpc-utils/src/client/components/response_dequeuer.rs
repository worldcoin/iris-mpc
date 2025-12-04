use async_trait::async_trait;

use super::super::typeset::{ClientError, ProcessRequestBatch, RequestBatch};
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
}

#[async_trait]
impl ProcessRequestBatch for ResponseDequeuer {
    async fn process_batch(&mut self, batch: &RequestBatch) -> Result<(), ClientError> {
        // TODO: dequeue system responses until the entire
        // batch is correlated or a timeout occurs.
        for _request in batch.requests() {
            self.aws_client.sqs_receive_message().await?;
        }

        Ok(())
    }
}
