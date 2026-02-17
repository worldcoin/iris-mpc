use async_trait::async_trait;

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::super::typeset::{
    ProcessRequestBatch, Request, RequestBatch, RequestPayload, RequestStatus, ServiceClientError,
};
use crate::aws::{types::SnsMessageInfo, AwsClient};

const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

/// Enqueues system requests upon network ingress queues.
pub(crate) struct RequestEnqueuer {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

impl RequestEnqueuer {
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }

    /// Publishes a deletion request for a single serial ID during cleanup.
    pub(crate) async fn publish_deletion(
        &self,
        serial_id: IrisSerialId,
    ) -> Result<(), ServiceClientError> {
        let payload =
            RequestPayload::IdentityDeletion(smpc_request::IdentityDeletionRequest { serial_id });
        let sns_msg_info = SnsMessageInfo::from(payload);
        self.aws_client
            .sns_publish_json(sns_msg_info)
            .await
            .map_err(ServiceClientError::AwsServiceError)
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
                request.set_status(RequestStatus::Enqueued);
            }
        }

        Ok(())
    }
}

impl From<&Request> for RequestPayload {
    fn from(request: &Request) -> Self {
        match request {
            Request::IdentityDeletion { parent, .. } => {
                Self::IdentityDeletion(smpc_request::IdentityDeletionRequest {
                    serial_id: parent
                        .get_serial_id()
                        .expect("response not received for parent request"),
                })
            }
            Request::Reauthorization {
                reauth_id, parent, ..
            } => Self::Reauthorization(smpc_request::ReAuthRequest {
                batch_size: Some(1),
                reauth_id: reauth_id.to_string(),
                s3_key: reauth_id.to_string(),
                serial_id: parent
                    .get_serial_id()
                    .expect("response not received for parent request"),
                skip_persistence: None,
                use_or_rule: false,
            }),
            Request::ResetCheck { reset_id, .. } => {
                Self::ResetCheck(smpc_request::ResetCheckRequest {
                    batch_size: Some(1),
                    reset_id: reset_id.to_string(),
                    s3_key: reset_id.to_string(),
                })
            }
            Request::ResetUpdate {
                reset_id, parent, ..
            } => Self::ResetUpdate(smpc_request::ResetUpdateRequest {
                reset_id: reset_id.to_string(),
                s3_key: reset_id.to_string(),
                serial_id: parent
                    .get_serial_id()
                    .expect("response not received for parent request"),
            }),
            Request::Uniqueness { signup_id, .. } => {
                Self::Uniqueness(smpc_request::UniquenessRequest {
                    batch_size: Some(1),
                    s3_key: signup_id.to_string(),
                    signup_id: signup_id.to_string(),
                    or_rule_serial_ids: None,
                    skip_persistence: None,
                    full_face_mirror_attacks_detection_enabled: Some(true),
                    disable_anonymized_stats: None,
                })
            }
        }
    }
}

impl From<&Request> for SnsMessageInfo {
    fn from(request: &Request) -> Self {
        Self::from(RequestPayload::from(request))
    }
}

impl From<RequestPayload> for SnsMessageInfo {
    fn from(body: RequestPayload) -> Self {
        match body {
            RequestPayload::IdentityDeletion(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE,
                &body,
            ),
            RequestPayload::Reauthorization(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::REAUTH_MESSAGE_TYPE,
                &body,
            ),
            RequestPayload::ResetCheck(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RESET_CHECK_MESSAGE_TYPE,
                &body,
            ),
            RequestPayload::ResetUpdate(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::RESET_UPDATE_MESSAGE_TYPE,
                &body,
            ),
            RequestPayload::Uniqueness(body) => Self::new(
                ENROLLMENT_REQUEST_TYPE,
                smpc_request::UNIQUENESS_MESSAGE_TYPE,
                &body,
            ),
        }
    }
}
