use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{UniquenessRequest, UNIQUENESS_MESSAGE_TYPE};
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::super::{
    errors::ServiceClientError,
    traits::{ComponentInitializer, RequestBatchProcesser},
    types::{RequestBatch, RequestData},
};
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
}

#[async_trait]
impl ComponentInitializer for RequestEnqueuer {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        match self.aws_client.set_public_keyset().await {
            Ok(_) => Ok(()),
            Err(e) => Err(ServiceClientError::ComponentInitialisationError(
                e.to_string(),
            )),
        }
    }
}

#[async_trait]
impl RequestBatchProcesser for RequestEnqueuer {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ServiceClientError> {
        for request in batch.requests() {
            match request.data() {
                RequestData::Uniqueness { shares } => {
                    match self.enqueue_uniqueness_request(request, shares).await {
                        Ok(_) => (),
                        Err(e) => {
                            return Err(ServiceClientError::EnqueueUniquenessRequestError(
                                e.to_string(),
                            ))
                        }
                    }
                }
                _ => {
                    return Err(ServiceClientError::UnsupportedRequestType(
                        request.data().to_string(),
                    ))
                }
            }
        }

        Ok(())
    }
}

impl RequestEnqueuer {
    /// Enqueues a uniqueness request.  This is a two stage process as encrypted shares are first
    /// uploaded to AWS S3 prior to actual enqueuing.
    async fn enqueue_uniqueness_request(
        &self,
        request: &Request,
        shares: &BothEyes<IrisCodeAndMaskShares>,
    ) -> Result<(), ServiceClientError> {
        // Step 0: Set sign-up id so that it matches request id.
        let signup_id = request.identifier();

        // Step 1: Upload encrypted shares to S3.
        let [[l_code, l_mask], [r_code, r_mask]] = shares.clone();
        let shares = create_iris_code_party_shares(*signup_id, l_code, l_mask, r_code, r_mask);
        let s3_key = match self
            .aws_client
            .encrypt_and_upload_iris_shares(&shares)
            .await
        {
            Ok(s3_key) => {
                tracing::info!("{} :: Shares encrypted and uploaded to S3", request);
                s3_key
            }
            Err(e) => return Err(ServiceClientError::AwsServiceError(e)),
        };

        // Step 2: Enqueue system request.
        let sns_message_type = UNIQUENESS_MESSAGE_TYPE;
        let sns_message_group_id = ENROLLMENT_REQUEST_TYPE;
        let sns_message_body = UniquenessRequest {
            batch_size: Some(1),
            signup_id: shares.signup_id.clone(),
            s3_key,
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
        };
        match self
            .aws_client
            .sns_publish::<UniquenessRequest>(
                sns_message_type,
                sns_message_group_id,
                sns_message_body,
            )
            .await
        {
            Ok(()) => {
                tracing::info!("{} :: Enqueued to AWS-SNS topic", request);
                Ok(())
            }
            Err(e) => Err(ServiceClientError::AwsServiceError(e)),
        }
    }
}
