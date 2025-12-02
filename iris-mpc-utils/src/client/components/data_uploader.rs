use async_trait::async_trait;
use rand::{CryptoRng, Rng};
use uuid::Uuid;

use super::super::typeset::{ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch};
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
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ClientError> {
        for request in batch.requests() {
            match request {
                Request::IdentityDeletion { signup_id, .. } => {
                    self.upload_iris_shares(signup_id).await?;
                }
                Request::Reauthorization {
                    reauth_id,
                    signup_id,
                    ..
                } => {
                    self.upload_iris_shares(reauth_id).await?;
                    self.upload_iris_shares(signup_id).await?;
                }
                Request::ResetCheck { reset_id, .. } => {
                    self.upload_iris_shares(reset_id).await?;
                }
                Request::ResetUpdate { reset_id, .. } => {
                    self.upload_iris_shares(reset_id).await?;
                }
                Request::Uniqueness { signup_id, .. } => {
                    self.upload_iris_shares(signup_id).await?;
                }
            }
        }

        Ok(())
    }
}

impl<R: Rng + CryptoRng + Send> DataUploader<R> {
    async fn upload_iris_shares(&mut self, identifier: &Uuid) -> Result<(), ClientError> {
        let shares = generate_iris_shares(self.rng_mut());

        self.aws_client
            .s3_upload_iris_shares(identifier, &shares)
            .await
            .map_err(ClientError::AwsServiceError)?;

        Ok(())
    }
}
