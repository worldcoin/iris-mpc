use async_trait::async_trait;

use super::super::typeset::{
    ClientError, Initialize, ProcessRequestBatch, Request, RequestBatch, RequestData,
};
use crate::aws::AwsClient;

/// A component responsible for uploading data to AWS services in advance of request processing.
#[derive(Debug)]
pub struct DataUploader {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,
}

impl DataUploader {
    /// Constructor.
    pub fn new(aws_client: AwsClient) -> Self {
        Self { aws_client }
    }
}

#[async_trait]
impl Initialize for DataUploader {
    async fn init(&mut self) -> Result<(), ClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ClientError::AwsServiceError)
    }
}

#[async_trait]
impl ProcessRequestBatch for DataUploader {
    async fn process_batch(&self, batch: &RequestBatch) -> Result<(), ClientError> {
        for request in batch.requests() {
            match request.data() {
                RequestData::IdentityDeletion { .. } => {
                    self.upload_identity_deletion(request).await?;
                }
                RequestData::Reauthorization { .. } => {
                    self.upload_reauthorization(request).await?;
                }
                RequestData::ResetCheck { .. } => {
                    println!("Upload ResetCheck data");
                }
                RequestData::ResetUpdate { .. } => {
                    println!("Upload ResetUpdate data");
                }
                RequestData::Uniqueness { .. } => {
                    self.upload_uniqueness(request).await?;
                }
            }
        }

        Ok(())
    }
}

impl DataUploader {
    async fn upload_identity_deletion(&self, request: &Request) -> Result<(), ClientError> {
        // Destructure generated data.
        let (signup_id, signup_shares) = match request.data() {
            RequestData::IdentityDeletion {
                signup_id,
                signup_shares,
                ..
            } => (signup_id, signup_shares),
            _ => unreachable!(),
        };

        // Upload to AWS-S3.
        self.aws_client
            .s3_upload_iris_shares(signup_id, signup_shares)
            .await
            .map(|_| ())
            .map_err(ClientError::AwsServiceError)
    }

    async fn upload_uniqueness(&self, request: &Request) -> Result<(), ClientError> {
        // Destructure generated data.
        let (signup_id, signup_shares) = match request.data() {
            RequestData::Uniqueness {
                signup_id,
                signup_shares,
                ..
            } => (signup_id, signup_shares),
            _ => unreachable!(),
        };

        // Upload to AWS-S3.
        self.aws_client
            .s3_upload_iris_shares(signup_id, signup_shares)
            .await
            .map(|_| ())
            .map_err(ClientError::AwsServiceError)
    }

    async fn upload_reauthorization(&self, request: &Request) -> Result<(), ClientError> {
        // Destructure generated data.
        let (reauthorisation_id, reauthorisation_shares, signup_id, signup_shares) =
            match request.data() {
                RequestData::Reauthorization {
                    reauthorisation_id,
                    reauthorisation_shares,
                    signup_id,
                    signup_shares,
                } => (
                    reauthorisation_id,
                    reauthorisation_shares,
                    signup_id,
                    signup_shares,
                ),
                _ => unreachable!(),
            };

        // Upload to AWS-S3.
        self.aws_client
            .s3_upload_iris_shares(signup_id, signup_shares)
            .await
            .map_err(ClientError::AwsServiceError)?;
        self.aws_client
            .s3_upload_iris_shares(reauthorisation_id, reauthorisation_shares)
            .await
            .map_err(ClientError::AwsServiceError)?;

        Ok(())
    }
}
