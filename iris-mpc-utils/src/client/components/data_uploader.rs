use async_trait::async_trait;
use futures;
use rand::{CryptoRng, Rng};

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
    async fn process_batch(&mut self, batch: &RequestBatch) -> Result<(), ClientError> {
        // Set shares to be uploaded.
        let mut shares = Vec::new();
        for request in batch.requests() {
            match request {
                Request::IdentityDeletion {
                    known_iris_serial_id,
                    signup_id,
                    ..
                } => {
                    if known_iris_serial_id.is_none() {
                        shares.push((signup_id, generate_iris_shares(self.rng_mut())));
                    }
                }
                Request::Reauthorization {
                    known_iris_serial_id,
                    reauth_id,
                    signup_id,
                    ..
                } => {
                    shares.push((reauth_id, generate_iris_shares(self.rng_mut())));
                    if known_iris_serial_id.is_none() {
                        shares.push((signup_id, generate_iris_shares(self.rng_mut())));
                    }
                }
                Request::ResetCheck {
                    known_iris_serial_id,
                    reset_id,
                    signup_id,
                    ..
                } => {
                    shares.push((reset_id, generate_iris_shares(self.rng_mut())));
                    if known_iris_serial_id.is_none() {
                        shares.push((signup_id, generate_iris_shares(self.rng_mut())));
                    }
                }
                Request::ResetUpdate {
                    known_iris_serial_id,
                    reset_id,
                    signup_id,
                    ..
                } => {
                    shares.push((reset_id, generate_iris_shares(self.rng_mut())));
                    if known_iris_serial_id.is_none() {
                        shares.push((signup_id, generate_iris_shares(self.rng_mut())));
                    }
                }
                Request::Uniqueness { signup_id, .. } => {
                    shares.push((signup_id, generate_iris_shares(self.rng_mut())));
                }
            }
        }

        // Set upload tasks.
        let aws_client = &self.aws_client;
        let tasks: Vec<_> = shares
            .iter()
            .map(|(identifier, shares)| async move {
                aws_client
                    .s3_upload_iris_shares(identifier, shares)
                    .await
                    .map_err(ClientError::AwsServiceError)
            })
            .collect();

        // Execute tasks in parallel.
        futures::future::try_join_all(tasks).await?;

        Ok(())
    }
}
