use async_trait::async_trait;

use super::types::{Batch, RequestDispatcher};
use crate::types::NetServiceClients;

/// A dispatcher that dispatches SMPC service requests to an AWS service.
#[derive(Debug)]
pub struct AwsRequestDispatcher {
    /// Associated network wide AWS service clients.
    #[allow(dead_code)]
    net_aws_clients: NetServiceClients,
}

impl AwsRequestDispatcher {
    pub fn new(net_aws_clients: NetServiceClients) -> Self {
        Self { net_aws_clients }
    }
}

#[async_trait]
impl RequestDispatcher for AwsRequestDispatcher {
    async fn dispatch(&self, batch: Batch) {
        for payload in batch.requests() {
            println!("TODO: dispatch {} :: {:?}", batch.batch_idx(), payload);
        }
    }
}
