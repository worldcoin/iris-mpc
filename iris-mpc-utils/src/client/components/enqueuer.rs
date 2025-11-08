use uuid::Uuid;

use iris_mpc_common::helpers::smpc_request::UniquenessRequest;

use super::super::types::{RequestBatch, RequestData};
use crate::{
    aws::{create_iris_code_party_shares, NetAwsClient},
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

    /// Enqueues a uniqueness request.  This is a two stage process as encrypted shares are first
    /// uploaded to AWS S3 prior to actual enqueuing.
    async fn enqueue_uniqueness_request(&self, request: &Request, data: &RequestDataUniqueness) {
        // Step 0: Set signup identifier.
        let signup_id = Uuid::new_v4();

        // Step 1: Set S3 shares.
        let [[l_code, l_mask], [r_code, r_mask]] = data.shares().clone();
        let shares = create_iris_code_party_shares(
            l_code,
            l_mask,
            r_code,
            r_mask,
            signup_id.to_string().into(),
        );
        println!(
            "{} :: S3 shares created :: signup_id={}",
            request, signup_id
        );

        // Step 2: Upload encrypted shares to an S3 bucket.
        let s3_bucket = match self.net_aws_client[0]
            .upload_iris_party_shares(&shares, &self.encryption_keys())
            .await
        {
            Err(report) => {
                panic!("{} :: {}", request, report);
            }
            Ok(s3_bucket) => {
                println!("{} :: S3 shares uploaded :: {}", request, s3_bucket);
                s3_bucket
            }
        };

        // Step 3: Set request payload.
        // TODO: use batch size ando ther fields from request
        let _request_payload = UniquenessRequest {
            batch_size: Some(1),
            signup_id: shares.signup_id.clone(),
            s3_key: s3_bucket,
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
        };
        println!("{} :: Uniqueness payload instantatiated", request);

        // Step 4: Enqueue on AWS SNS.

        println!("{} :: Enqueueing", request);
    }
}
