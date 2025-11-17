use async_trait::async_trait;
use serde::Serialize;

use iris_mpc_common::helpers::smpc_request::{
    IdentityDeletionRequest, ReAuthRequest, ResetCheckRequest, ResetUpdateRequest,
    UniquenessRequest, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use super::super::{
    errors::ServiceClientError,
    traits::{ComponentInitializer, ProcessRequestBatch},
    types::{RequestBatch, RequestData},
};
use crate::{
    aws::{
        create_iris_code_party_shares, create_iris_party_shares_for_s3,
        types::{S3ObjectInfo, SnsMessageInfo},
        AwsClient,
    },
    client::types::Request,
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
impl ProcessRequestBatch for RequestEnqueuer {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ServiceClientError> {
        for request in batch.requests() {
            match request.data() {
                RequestData::IdentityDeletion { .. } => {
                    self.enqueue_identity_deletion(request).await
                }
                RequestData::Reauthorization { .. } => self.enqueue_reauthorization(request).await,
                RequestData::ResetCheck { .. } => self.enqueue_reset_check(request).await,
                RequestData::ResetUpdate { .. } => self.enqueue_reset_update(request).await,
                RequestData::Uniqueness { .. } => self.enqueue_uniqueness_request(request).await,
            }
            .map_err(|e| ServiceClientError::EnqueueRequestError(e.to_string()))?;
            tracing::info!("{} :: Enqueued to AWS-SNS topic", request);
        }

        Ok(())
    }
}

impl RequestEnqueuer {
    /// Enqueues a system request by dispatching a notification to AWS-SNS.
    async fn enqueue<T>(&self, data: T) -> Result<(), ServiceClientError>
    where
        T: Sized + Serialize,
        SnsMessageInfo<T>: From<T>,
    {
        match self
            .aws_client
            .sns_publish_json(SnsMessageInfo::from(data))
            .await
        {
            Ok(()) => Ok(()),
            Err(e) => Err(ServiceClientError::AwsServiceError(e)),
        }
    }

    /// Enqueues a uniqueness request.  This is a two stage process as encrypted shares are first
    /// uploaded to AWS S3 prior to actual enqueuing.
    async fn enqueue_uniqueness_request(
        &self,
        request: &Request,
    ) -> Result<(), ServiceClientError> {
        // Destructure generated data.
        let shares = match request.data() {
            RequestData::Uniqueness { shares } => shares,
            _ => unreachable!(),
        };

        // Set signup id.
        let signup_id = request.identifier().clone();

        // Set AWS-S3 JSON compatible shares.
        let [[l_code, l_mask], [r_code, r_mask]] = shares.clone();
        let shares = create_iris_party_shares_for_s3(
            &create_iris_code_party_shares(
                signup_id.clone(),
                l_code.to_owned(),
                l_mask.to_owned(),
                r_code.to_owned(),
                r_mask.to_owned(),
            ),
            &self.aws_client.public_keyset(),
        );

        // Upload to AWS-S3.
        let s3_bucket = self.aws_client.config().s3_request_bucket_name();
        let s3_key = signup_id.to_string();
        let s3_obj_info = S3ObjectInfo::new_from_jsonic(&s3_bucket, &s3_key, &shares);
        match self.aws_client.s3_put_object(&s3_obj_info).await {
            Ok(_) => {
                tracing::info!("{} :: Shares encrypted and uploaded to S3", request);
            }
            Err(e) => return Err(ServiceClientError::AwsServiceError(e)),
        };

        // Enqueue to AWS-SNS.
        self.enqueue(UniquenessRequest {
            batch_size: Some(1),
            signup_id: signup_id.to_string(),
            s3_key,
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
        })
        .await
    }

    /// Enqueues an identity deletion request.
    async fn enqueue_identity_deletion(
        &self,
        _request: &Request,
    ) -> Result<(), ServiceClientError> {
        // TODO: hydrate correctly.
        self.enqueue(IdentityDeletionRequest {
            serial_id: u32::MIN,
        })
        .await
    }

    /// Enqueues a reauthorization request.
    async fn enqueue_reauthorization(&self, _request: &Request) -> Result<(), ServiceClientError> {
        // TODO: hydrate correctly.
        self.enqueue(ReAuthRequest {
            reauth_id: "reauth_id".to_string(),
            batch_size: None,
            s3_key: "s3_key".to_string(),
            serial_id: u32::MIN,
            use_or_rule: false,
        })
        .await
    }

    /// Enqueues a reset check request.
    async fn enqueue_reset_check(&self, _request: &Request) -> Result<(), ServiceClientError> {
        // TODO: hydrate correctly.
        self.enqueue(ResetCheckRequest {
            reset_id: "reset_id".to_string(),
            batch_size: None,
            s3_key: "s3_key".to_string(),
        })
        .await
    }

    /// Enqueues a reset update request.
    async fn enqueue_reset_update(&self, _request: &Request) -> Result<(), ServiceClientError> {
        // TODO: hydrate correctly.
        self.enqueue(ResetUpdateRequest {
            reset_id: "reset_id".to_string(),
            serial_id: u32::MIN,
            s3_key: "s3_key".to_string(),
        })
        .await
    }
}

impl From<IdentityDeletionRequest> for SnsMessageInfo<IdentityDeletionRequest> {
    fn from(body: IdentityDeletionRequest) -> Self {
        SnsMessageInfo::<IdentityDeletionRequest>::new(
            body,
            String::from(ENROLLMENT_REQUEST_TYPE),
            String::from(IDENTITY_DELETION_MESSAGE_TYPE),
        )
    }
}

impl From<ReAuthRequest> for SnsMessageInfo<ReAuthRequest> {
    fn from(body: ReAuthRequest) -> Self {
        SnsMessageInfo::<ReAuthRequest>::new(
            body,
            String::from(ENROLLMENT_REQUEST_TYPE),
            String::from(REAUTH_MESSAGE_TYPE),
        )
    }
}

impl From<ResetCheckRequest> for SnsMessageInfo<ResetCheckRequest> {
    fn from(body: ResetCheckRequest) -> Self {
        SnsMessageInfo::<ResetCheckRequest>::new(
            body,
            String::from(ENROLLMENT_REQUEST_TYPE),
            String::from(RESET_CHECK_MESSAGE_TYPE),
        )
    }
}

impl From<ResetUpdateRequest> for SnsMessageInfo<ResetUpdateRequest> {
    fn from(body: ResetUpdateRequest) -> Self {
        SnsMessageInfo::<ResetUpdateRequest>::new(
            body,
            String::from(ENROLLMENT_REQUEST_TYPE),
            String::from(RESET_UPDATE_MESSAGE_TYPE),
        )
    }
}

impl From<UniquenessRequest> for SnsMessageInfo<UniquenessRequest> {
    fn from(body: UniquenessRequest) -> Self {
        SnsMessageInfo::<UniquenessRequest>::new(
            body,
            String::from(ENROLLMENT_REQUEST_TYPE),
            String::from(UNIQUENESS_MESSAGE_TYPE),
        )
    }
}
