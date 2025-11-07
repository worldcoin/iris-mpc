use super::types::{RequestBatch, RequestData};
use crate::{
    aws::NetAwsClient,
    client::types::{Request, RequestDataUniqueness},
};

/// Dispatches SMPC service requests via an AWS service.
#[derive(Debug)]
pub struct RequestDispatcher {
    /// Associated network wide AWS service clients.
    #[allow(dead_code)]
    net_aws_client: NetAwsClient,
}

impl RequestDispatcher {
    pub fn new(net_aws_client: NetAwsClient) -> Self {
        Self { net_aws_client }
    }

    pub async fn dispatch(&self, batch: RequestBatch) {
        for request in batch.requests() {
            match request.data() {
                RequestData::Uniqueness(data) => {
                    self.dispatch_uniqueness_request(request, data).await;
                }
                _ => panic!("Unsupported request type"),
            }
        }
    }

    async fn dispatch_uniqueness_request(&self, request: &Request, _data: &RequestDataUniqueness) {
        println!("Dispatching request: {}", request,);
    }
}
