use super::super::types::{RequestBatch, RequestData};
use crate::{
    aws::NetAwsClient,
    client::types::{Request, RequestDataUniqueness},
};

/// A component responsible for enqueuing system requests upon network ingress queues.
#[derive(Debug)]
pub struct RequestEnqueuer {
    /// A client for interacting with any node's AWS service.
    #[allow(dead_code)]
    net_aws_client: NetAwsClient,
}

impl RequestEnqueuer {
    pub fn new(net_aws_client: NetAwsClient) -> Self {
        Self { net_aws_client }
    }

    /// Enqueues a batch of system requests upon each node's ingress queue.
    pub async fn enqueue(&self, batch: &RequestBatch) {
        for request in batch.requests() {
            match request.data() {
                RequestData::Uniqueness(data) => {
                    self.enqueue_uniqueness_request(request, data).await;
                }
                _ => panic!("Unsupported request type"),
            }
        }
    }

    async fn enqueue_uniqueness_request(&self, request: &Request, _data: &RequestDataUniqueness) {
        println!("Dispatching request: {}", request,);
    }
}
