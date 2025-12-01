use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, ProcessRequestBatch, Request, RequestBatch, RequestData, RequestMessageBody,
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
            let request_body = self.get_message_body(request).await?;
            self.aws_client
                .sns_publish_json(SnsMessageInfo::from(request_body))
                .await
                .map_err(ClientError::AwsServiceError)?;
            tracing::info!("{}: Published to AWS-SNS", request);
        }

        Ok(())
    }
}

impl RequestEnqueuer {
    /// Returns body of message to be enqueued.
    async fn get_message_body(&self, request: &Request) -> Result<RequestMessageBody, ClientError> {
        match request.data() {
            RequestData::IdentityDeletion { .. } => self.get_identity_deletion(request).await,
            RequestData::Reauthorization { .. } => self.get_reauthorization(request).await,
            RequestData::ResetCheck { .. } => self.get_reset_check(request).await,
            RequestData::ResetUpdate { .. } => self.get_reset_update(request).await,
            RequestData::Uniqueness { .. } => self.get_uniqueness_request(request).await,
        }
    }

    async fn get_identity_deletion(
        &self,
        _request: &Request,
    ) -> Result<RequestMessageBody, ClientError> {
        // Set body payload.
        // TODO: get serial id from correlated uniqueness request.
        let payload = IdentityDeletionRequest { serial_id: 2 };

        Ok(RequestMessageBody::IdentityDeletion(payload))
    }

    async fn get_reauthorization(
        &self,
        request: &Request,
    ) -> Result<RequestMessageBody, ClientError> {
        // Destructure generated data.
        let reauthorisation_id = match request.data() {
            RequestData::Reauthorization {
                reauthorisation_id, ..
            } => reauthorisation_id,
            _ => unreachable!(),
        };

        // Set body payload.
        let payload = ReAuthRequest {
            batch_size: Some(1),
            reauth_id: reauthorisation_id.to_string(),
            s3_key: reauthorisation_id.to_string(),
            serial_id: 1,
            use_or_rule: bool::default(),
        };

        Ok(RequestMessageBody::Reauthorization(payload))
    }

    async fn get_reset_check(&self, _request: &Request) -> Result<RequestMessageBody, ClientError> {
        unimplemented!()
    }

    async fn get_reset_update(
        &self,
        _request: &Request,
    ) -> Result<RequestMessageBody, ClientError> {
        unimplemented!()
    }

    async fn get_uniqueness_request(
        &self,
        request: &Request,
    ) -> Result<RequestMessageBody, ClientError> {
        // Destructure generated data.
        let signup_id = match request.data() {
            RequestData::Uniqueness { signup_id, .. } => signup_id,
            _ => unreachable!(),
        };

        // Set body payload.
        let payload = UniquenessRequest {
            batch_size: Some(1),
            signup_id: signup_id.to_string(),
            s3_key: signup_id.to_string(),
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
        };

        Ok(RequestMessageBody::Uniqueness(payload))
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
