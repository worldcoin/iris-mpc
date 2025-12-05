use async_trait::async_trait;

use super::super::typeset::{ClientError, Initialize, ProcessRequestBatch, RequestBatch};
use crate::aws::AwsClient;

/// A component responsible for correlating system requests with system responses.
#[derive(Debug)]
pub struct ResponseCorrelator {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

impl ResponseCorrelator {
    /// Constructor.
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    #[allow(dead_code)]
    pub async fn correlate(&self, batch: &RequestBatch) {
        tracing::info!(
            "TODO: correlate enqueued requests with dequeued responses: {}",
            batch
        );
    }
}

#[async_trait]
impl Initialize for ResponseCorrelator {
    async fn init(&mut self) -> Result<(), ClientError> {
        self.aws_client
            .sqs_purge_response_queue()
            .await
            .map_err(ClientError::AwsServiceError)
    }
}

#[async_trait]
impl ProcessRequestBatch for ResponseCorrelator {
    async fn process_batch(&mut self, _batch: &mut RequestBatch) -> Result<(), ClientError> {
        // TODO: perform correlation of requests with responses
        Ok(())
    }
}
