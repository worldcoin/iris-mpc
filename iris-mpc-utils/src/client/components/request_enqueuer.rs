use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
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
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ClientError> {
        // Execute enqueue tasks in parallel.
        let tasks: Vec<_> = batch
            .requests()
            .iter()
            .enumerate()
            .filter(|(_, r)| r.can_enqueue())
            .map(|(idx, request)| {
                let aws_client = &self.aws_client;
                let sns_message_info = SnsMessageInfo::from(request);
                async move {
                    aws_client
                        .sns_publish_json(sns_message_info)
                        .await
                        .map_err(ClientError::AwsServiceError)?;
                    Ok::<usize, ClientError>(idx)
                }
            })
            .collect();
        let enqueued_indices = futures::future::try_join_all(tasks).await?;

        // Mark requests as enqueued after successful publish.
        for idx in enqueued_indices {
            if let Some(request) = batch.requests_mut().get_mut(idx) {
                request.set_status_enqueued();
            }
        }

        Ok(())
    }
}

impl From<&Request> for RequestMessageBody {
    fn from(request: &Request) -> Self {
        // TODO: serial ID from correlated uniqueness response
        match request {
            Request::IdentityDeletion { serial_id, .. } => {
                Self::IdentityDeletion(IdentityDeletionRequest {
                    serial_id: serial_id.unwrap_or(1),
                })
            }
            Request::Reauthorization {
                reauth_id,
                serial_id,
                ..
            } => Self::Reauthorization(ReAuthRequest {
                batch_size: Some(1),
                reauth_id: reauth_id.to_string(),
                s3_key: reauth_id.to_string(),
                serial_id: serial_id.unwrap_or(1),
                use_or_rule: false,
            }),
            Request::ResetCheck { reset_check_id, .. } => Self::ResetCheck(ResetCheckRequest {
                batch_size: Some(1),
                reset_id: reset_check_id.to_string(),
                s3_key: reset_check_id.to_string(),
            }),
            Request::ResetUpdate {
                reset_update_id,
                serial_id,
                ..
            } => Self::ResetUpdate(ResetUpdateRequest {
                reset_id: reset_update_id.to_string(),
                s3_key: reset_update_id.to_string(),
                serial_id: serial_id.unwrap_or(1),
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
        Self::from(RequestMessageBody::from(request))
    }
}

impl From<RequestMessageBody> for SnsMessageInfo {
    fn from(body: RequestMessageBody) -> Self {
        match body {
            RequestMessageBody::IdentityDeletion(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                IDENTITY_DELETION_MESSAGE_TYPE,
                &body,
            ),
            RequestMessageBody::Reauthorization(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, REAUTH_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::ResetCheck(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, RESET_CHECK_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::ResetUpdate(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, RESET_UPDATE_MESSAGE_TYPE, &body)
            }
            RequestMessageBody::Uniqueness(body) => {
                Self::new(ENROLLMENT_REQUEST_TYPE, UNIQUENESS_MESSAGE_TYPE, &body)
            }
        }
    }
}
