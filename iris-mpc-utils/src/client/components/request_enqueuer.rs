use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ProcessRequestBatch, Request, RequestBatch, RequestBody, RequestStatus, ServiceClientError,
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
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ServiceClientError> {
        // Set enqueue tasks.
        let tasks: Vec<_> = batch
            .requests()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_enqueueable())
            .map(|(idx, request)| {
                let aws_client = &self.aws_client;
                let sns_msg_info = SnsMessageInfo::from(request);
                async move {
                    aws_client
                        .sns_publish_json(sns_msg_info)
                        .await
                        .map_err(ServiceClientError::AwsServiceError)?;
                    Ok::<usize, ServiceClientError>(idx)
                }
            })
            .collect();

        // Enqueue & mark requests as enqueued.
        for idx in futures::future::try_join_all(tasks).await? {
            if let Some(request) = batch.requests_mut().get_mut(idx) {
                request.set_status(RequestStatus::new_enqueued());
            }
        }

        Ok(())
    }
}

impl From<&Request> for RequestBody {
    fn from(request: &Request) -> Self {
        match request {
            Request::IdentityDeletion {
                uniqueness_serial_id,
                ..
            } => Self::IdentityDeletion(IdentityDeletionRequest {
                serial_id: uniqueness_serial_id.unwrap_or(1),
            }),
            Request::Reauthorization {
                reauth_id,
                uniqueness_serial_id,
                ..
            } => Self::Reauthorization(ReAuthRequest {
                batch_size: Some(1),
                reauth_id: reauth_id.to_string(),
                s3_key: reauth_id.to_string(),
                serial_id: uniqueness_serial_id.unwrap_or(1),
                use_or_rule: false,
            }),
            Request::ResetCheck { reset_id, .. } => Self::ResetCheck(ResetCheckRequest {
                batch_size: Some(1),
                reset_id: reset_id.to_string(),
                s3_key: reset_id.to_string(),
            }),
            Request::ResetUpdate {
                reset_id,
                uniqueness_serial_id,
                ..
            } => Self::ResetUpdate(ResetUpdateRequest {
                reset_id: reset_id.to_string(),
                s3_key: reset_id.to_string(),
                serial_id: uniqueness_serial_id.unwrap_or(1),
            }),
            Request::Uniqueness { signup_id, .. } => Self::Uniqueness(UniquenessRequest {
                batch_size: Some(1),
                s3_key: signup_id.to_string(),
                signup_id: signup_id.to_string(),
                or_rule_serial_ids: None,
                skip_persistence: None,
                full_face_mirror_attacks_detection_enabled: Some(true),
                disable_anonymized_stats: None,
            }),
        }
    }
}

impl From<&Request> for SnsMessageInfo {
    fn from(request: &Request) -> Self {
        Self::from(RequestBody::from(request))
    }
}

impl From<RequestBody> for SnsMessageInfo {
    fn from(body: RequestBody) -> Self {
        match body {
            RequestBody::IdentityDeletion(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                IDENTITY_DELETION_MESSAGE_TYPE,
                &body,
            ),
            RequestBody::Reauthorization(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, REAUTH_MESSAGE_TYPE, &body)
            }
            RequestBody::ResetCheck(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, RESET_CHECK_MESSAGE_TYPE, &body)
            }
            RequestBody::ResetUpdate(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, RESET_UPDATE_MESSAGE_TYPE, &body)
            }
            RequestBody::Uniqueness(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, UNIQUENESS_MESSAGE_TYPE, &body)
            }
        }
    }
}
