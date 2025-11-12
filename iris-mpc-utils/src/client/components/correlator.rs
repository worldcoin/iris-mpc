use std::fmt;

use super::super::{errors::ServiceClientError, types::RequestBatch};
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

    /// Initializer.
    pub async fn init(&self) -> Result<(), ServiceClientError> {
        tracing::info!("Initialising ...");
        match self
            .aws_client
            .sqs_purge_queue(self.aws_client.config().sqs_response_queue_url())
            .await
        {
            Ok(()) => {
                tracing::info!("Purged SQS response queue");
                Ok(())
            }
            Err(e) => Err(ServiceClientError::ComponentInitialisationError(
                e.to_string(),
            )),
        }
    }

    #[allow(dead_code)]
    pub async fn correlate(&self, batch: &RequestBatch) {
        tracing::info!(
            "TODO: correlate enqueued requests with dequeued responses: {}",
            batch
        );
    }
}

impl fmt::Display for ResponseCorrelator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ResponseCorrelator",)
    }
}
