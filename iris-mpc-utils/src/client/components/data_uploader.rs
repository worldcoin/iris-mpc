use async_trait::async_trait;
use rand::{CryptoRng, Rng};
use uuid::Uuid;

use super::super::typeset::{
    ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch, RequestData,
};
use crate::{
    aws::AwsClient, irises::generate_iris_code_and_mask_shares_both_eyes as generate_iris_shares,
};

/// A component responsible for uploading data to AWS services in advance of request processing.
#[derive(Debug)]
pub struct DataUploader<R: Rng + CryptoRng + Send> {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,

    /// Entropy source.
    rng: R,
}

impl<R: Rng + CryptoRng + Send> DataUploader<R> {
    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Constructor.
    pub fn new(aws_client: AwsClient, rng: R) -> Self {
        Self { aws_client, rng }
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + Send> Initialize for DataUploader<R> {
    async fn init(&mut self) -> Result<(), ClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ClientError::AwsServiceError)
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + Send> ProcessRequestBatch for DataUploader<R> {
    async fn process_batch(&mut self, batch: &RequestBatch) -> Result<(), ClientError> {
        for request in batch.requests() {
            match request.data() {
                RequestData::IdentityDeletion { .. } => {
                    self.upload_identity_deletion(request).await?;
                }
                RequestData::Reauthorization { .. } => {
                    self.upload_reauthorization(request).await?;
                }
                RequestData::ResetCheck { .. } => {
                    self.upload_reset_check(request).await?;
                }
                RequestData::ResetUpdate { .. } => {
                    self.upload_reset_update(request).await?;
                }
                RequestData::Uniqueness { .. } => {
                    self.upload_uniqueness(request).await?;
                }
            }
        }

        Ok(())
    }
}

impl<R: Rng + CryptoRng + Send> DataUploader<R> {
    async fn upload_identity_deletion(&mut self, request: &Request) -> Result<(), ClientError> {
        let signup_id = match request.data() {
            RequestData::IdentityDeletion { signup_id, .. } => signup_id,
            _ => unreachable!(),
        };

        self.upload_iris_shares(signup_id).await?;

        Ok(())
    }

    async fn upload_reset_check(&mut self, request: &Request) -> Result<(), ClientError> {
        let reset_id = match request.data() {
            RequestData::ResetCheck { reset_id, .. } => reset_id,
            _ => unreachable!(),
        };

        self.upload_iris_shares(reset_id).await?;

        Ok(())
    }

    async fn upload_reset_update(&mut self, request: &Request) -> Result<(), ClientError> {
        let reset_id = match request.data() {
            RequestData::ResetUpdate { reset_id, .. } => reset_id,
            _ => unreachable!(),
        };

        self.upload_iris_shares(reset_id).await?;

        Ok(())
    }

    async fn upload_uniqueness(&mut self, request: &Request) -> Result<(), ClientError> {
        let signup_id = match request.data() {
            RequestData::Uniqueness { signup_id, .. } => signup_id,
            _ => unreachable!(),
        };

        self.upload_iris_shares(signup_id).await?;

        Ok(())
    }

    async fn upload_reauthorization(&mut self, request: &Request) -> Result<(), ClientError> {
        let (reauthorisation_id, signup_id) = match request.data() {
            RequestData::Reauthorization {
                reauthorisation_id,
                signup_id,
                ..
            } => (reauthorisation_id, signup_id),
            _ => unreachable!(),
        };

        self.upload_iris_shares(reauthorisation_id).await?;
        self.upload_iris_shares(signup_id).await?;

        Ok(())
    }

    async fn upload_iris_shares(&mut self, identifier: &Uuid) -> Result<(), ClientError> {
        let shares = generate_iris_shares(self.rng_mut());

        self.aws_client
            .s3_upload_iris_shares(identifier, &shares)
            .await
            .map_err(ClientError::AwsServiceError)?;

        Ok(())
    }
}
