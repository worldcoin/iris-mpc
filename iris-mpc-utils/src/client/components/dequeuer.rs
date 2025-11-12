use std::fmt;

use super::super::{errors::ServiceClientError, types::RequestBatch};
use crate::aws::AwsClient;

/// A component responsible for dequeuing system responses from network egress queues.
#[derive(Debug)]
pub struct ResponseDequeuer {
    /// A client for interacting with system AWS services.
    #[allow(dead_code)]
    aws_client: AwsClient,
}

impl ResponseDequeuer {
    /// Constructor.
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    /// Dequeues system responses from network egress queues.
    #[allow(dead_code)]
    pub async fn dequeue(&self, _batch: &RequestBatch) -> Result<(), ServiceClientError> {
        unimplemented!()
    }
}

impl fmt::Display for ResponseDequeuer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ResponseDequeuer",)
    }
}
