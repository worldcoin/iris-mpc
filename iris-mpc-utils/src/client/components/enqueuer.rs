use iris_mpc_common::helpers::smpc_request::{UniquenessRequest, UNIQUENESS_MESSAGE_TYPE};
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::super::types::{RequestBatch, RequestData};
use crate::{
    aws::{create_iris_code_party_shares, AwsClient},
    client::types::Request,
    types::IrisCodeAndMaskShares,
};

const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

/// A component responsible for enqueuing system requests upon network ingress queues.
#[derive(Debug)]
pub struct RequestEnqueuer {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

impl RequestEnqueuer {
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    /// Enqueues a batch of system requests upon each node's ingress queue.
    pub async fn enqueue(&self, batch: &RequestBatch) {
        for request in batch.requests() {
            match request.data() {
                RequestData::Uniqueness { shares } => {
                    self.enqueue_uniqueness_request(&request, shares).await
                }
                _ => panic!("Unsupported request type"),
            }
        }
    }

    /// Enqueues a uniqueness request.  This is a two stage process as encrypted shares are first
    /// uploaded to AWS S3 prior to actual enqueuing.
    async fn enqueue_uniqueness_request(
        &self,
        request: &Request,
        shares: &BothEyes<IrisCodeAndMaskShares>,
    ) {
        // Step 1: Set encrypted shares.
        let signup_id = request.identifier().clone();
        let [[l_code, l_mask], [r_code, r_mask]] = shares.clone();
        let shares = create_iris_code_party_shares(signup_id, l_code, l_mask, r_code, r_mask);

        // Step 2: Upload encrypted shares to S3.
        let s3_key = match self
            .aws_client
            .encrypt_and_upload_iris_shares(&shares)
            .await
        {
            Err(report) => {
                panic!("{} :: {}", request, report);
            }
            Ok(s3_key) => {
                println!(
                    "{} :: Shares encrypted and uploaded to S3 -> {}",
                    request, s3_key
                );
                s3_key
            }
        };

        // Step 3: Set system request payload.
        let payload = UniquenessRequest {
            batch_size: Some(1),
            signup_id: shares.signup_id.clone(),
            s3_key,
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
        };

        // Step 4: Enqueue system request.
        // TODO: handle enqueue error.
        self.aws_client
            .publish_to_sns::<UniquenessRequest>(
                UNIQUENESS_MESSAGE_TYPE,
                ENROLLMENT_REQUEST_TYPE,
                payload,
            )
            .await
            .unwrap();

        println!("{} :: request enqueued", request);
    }
}
