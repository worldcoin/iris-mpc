use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, ProcessRequestBatch, Request, RequestBatch, RequestMessageBody,
};
use crate::aws::{types::SnsMessageInfo, AwsClient};

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
impl ProcessRequestBatch for RequestEnqueuer {
    async fn process_batch(&mut self, batch: &RequestBatch) -> Result<(), ClientError> {
        for request in batch.requests() {
            self.aws_client
                .sns_publish_json(SnsMessageInfo::from(RequestMessageBody::from(request)))
                .await
                .map_err(ClientError::AwsServiceError)?;
            tracing::info!("{}: Published to AWS-SNS", request);
        }

        Ok(())
    }
}

impl From<&Request> for RequestMessageBody {
    fn from(request: &Request) -> Self {
        // TODO: ensure correlated data is available here.
        match request {
            Request::IdentityDeletion { .. } => {
                RequestMessageBody::IdentityDeletion(IdentityDeletionRequest { serial_id: 2 })
            }
            Request::Reauthorization { reauth_id, .. } => {
                RequestMessageBody::Reauthorization(ReAuthRequest {
                    batch_size: Some(1),
                    reauth_id: reauth_id.to_string(),
                    s3_key: reauth_id.to_string(),
                    serial_id: 1,
                    use_or_rule: bool::default(),
                })
            }
            Request::ResetCheck { .. } => unimplemented!(),
            Request::ResetUpdate { .. } => unimplemented!(),
            Request::Uniqueness { signup_id, .. } => {
                RequestMessageBody::Uniqueness(UniquenessRequest {
                    batch_size: Some(1),
                    signup_id: signup_id.to_string(),
                    s3_key: signup_id.to_string(),
                    or_rule_serial_ids: None,
                    skip_persistence: None,
                    full_face_mirror_attacks_detection_enabled: Some(true),
                    disable_anonymized_stats: None,
                })
            }
        }
    }
}

impl From<RequestMessageBody> for SnsMessageInfo {
    fn from(body: RequestMessageBody) -> Self {
        match body {
            RequestMessageBody::IdentityDeletion(body) => SnsMessageInfo::new(
                ENROLLMENT_REQUEST_TYPE,
                IDENTITY_DELETION_MESSAGE_TYPE,
                &body,
            ),
            RequestMessageBody::Reauthorization(body) => {
                SnsMessageInfo::new(ENROLLMENT_REQUEST_TYPE, REAUTH_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::ResetCheck(body) => {
                SnsMessageInfo::new(ENROLLMENT_REQUEST_TYPE, RESET_CHECK_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::ResetUpdate(body) => {
                SnsMessageInfo::new(ENROLLMENT_REQUEST_TYPE, RESET_UPDATE_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::Uniqueness(body) => {
                SnsMessageInfo::new(ENROLLMENT_REQUEST_TYPE, UNIQUENESS_MESSAGE_TYPE, &body)
            }
        }
    }
}
