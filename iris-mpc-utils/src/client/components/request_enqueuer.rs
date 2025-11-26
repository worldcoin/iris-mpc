use async_trait::async_trait;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE,
    REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};

use super::super::typeset::{
    ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch, RequestData,
    RequestMessageBody,
};
use crate::aws::{
    create_iris_code_party_shares, create_iris_party_shares_for_s3,
    types::{S3ObjectInfo, SnsMessageInfo},
    AwsClient,
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
impl Initialize for RequestEnqueuer {
    async fn init(&mut self) -> Result<(), ClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ClientError::AwsServiceError)
    }
}

#[async_trait]
impl ProcessRequestBatch for RequestEnqueuer {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ClientError> {
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
        let _shares = match request.data() {
            RequestData::Reauthorization { shares } => shares,
            _ => unreachable!(),
        };

        // Set body payload.
        let payload = ReAuthRequest {
            batch_size: None,
            reauth_id: String::from("reauth_id"),
            s3_key: String::from("s3_key"),
            serial_id: u32::default(),
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
        let shares = match request.data() {
            RequestData::Uniqueness { shares } => shares,
            _ => unreachable!(),
        };

        // Set AWS-S3 JSON compatible shares.  Signup id is derived from request id.
        let [[l_code, l_mask], [r_code, r_mask]] = shares.clone();
        let shares = create_iris_party_shares_for_s3(
            &create_iris_code_party_shares(
                *request.identifier(),
                l_code.to_owned(),
                l_mask.to_owned(),
                r_code.to_owned(),
                r_mask.to_owned(),
            ),
            &self.aws_client.public_keyset(),
        );

        // Upload to AWS-S3.
        let s3_obj_info = S3ObjectInfo::new(
            self.aws_client.config().s3_request_bucket_name(),
            &request.identifier().to_string(),
            &shares,
        );
        match self.aws_client.s3_put_object(s3_obj_info).await {
            Ok(_) => {
                tracing::info!("{} :: Shares encrypted and uploaded to S3", request);
            }
            Err(e) => return Err(ClientError::AwsServiceError(e)),
        };

        // Set body payload.
        let payload = UniquenessRequest {
            batch_size: Some(1),
            signup_id: request.identifier().to_string(),
            s3_key: request.identifier().to_string(),
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
