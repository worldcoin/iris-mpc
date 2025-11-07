use super::super::types::{RequestBatch, RequestData};
use crate::{
    aws::{create_iris_code_party_shares, create_iris_party_shares_for_s3, NetAwsClient},
    client::types::{Request, RequestDataUniqueness},
    types::NetEncryptionPublicKeys,
};

/// A component responsible for enqueuing system requests upon network ingress queues.
#[derive(Debug)]
pub struct RequestEnqueuer {
    /// A client for interacting with any node's AWS service.
    #[allow(dead_code)]
    net_aws_client: NetAwsClient,
}

impl RequestEnqueuer {
    fn encryption_keys(&self) -> &NetEncryptionPublicKeys {
        self.net_aws_client[0].encryption_keys()
    }

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

    async fn enqueue_uniqueness_request(&self, request: &Request, data: &RequestDataUniqueness) {
        // Step 1: Upload to an S3 bucket encrypted share set.
        let [[l_code, l_mask], [r_code, r_mask]] =
            data.iris_code_and_mask_shares_both_eyes().clone();
        let signup_id = None;
        let shares = create_iris_code_party_shares(l_code, l_mask, r_code, r_mask, signup_id);
        let _shares_s3 = create_iris_party_shares_for_s3(&shares, &self.encryption_keys());

        println!("Enqueueing {}", request);
    }
}
