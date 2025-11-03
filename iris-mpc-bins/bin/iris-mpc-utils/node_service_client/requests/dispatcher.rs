use async_trait::async_trait;

use iris_mpc_utils::types::NetServiceClients;

use super::types::{Batch, RequestDispatcher};

/// A dispatcher that dispatches SMPC service requests to an AWS service.
#[derive(Debug)]
pub struct AwsDispatcher {
    /// Associated network wide AWS service clients.
    net_aws_clients: NetServiceClients,
}

impl AwsDispatcher {
    pub fn new(net_aws_clients: NetServiceClients) -> Self {
        Self { net_aws_clients }
    }
}

#[async_trait]
impl RequestDispatcher for AwsDispatcher {
    async fn dispatch(&self, batch: Batch) {
        println!("TODO: dispatch request batch: {:?}", batch);
    }
}
