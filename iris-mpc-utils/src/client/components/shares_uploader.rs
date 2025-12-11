use async_trait::async_trait;
use futures;
use rand::{CryptoRng, Rng};

use super::super::typeset::{ClientError, Initialize, ProcessRequestBatch, RequestBatch};
use crate::{
    aws::AwsClient, irises::generate_iris_code_and_mask_shares_both_eyes as generate_iris_shares,
};

/// A component responsible for uploading Iris shares to AWS services
/// in advance of system request processing.
#[derive(Debug)]
pub struct SharesUploader<R: Rng + CryptoRng + Send> {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,

    /// Entropy source.
    rng: R,
}

impl<R: Rng + CryptoRng + Send> SharesUploader<R> {
    fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Constructor.
    pub fn new(aws_client: AwsClient, rng: R) -> Self {
        Self { aws_client, rng }
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + Send> Initialize for SharesUploader<R> {
    async fn init(&mut self) -> Result<(), ClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ClientError::AwsServiceError)
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + Send> ProcessRequestBatch for SharesUploader<R> {
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ClientError> {
        // Set shares to be uploaded.
        let shares: Vec<_> = batch
            .requests_mut()
            .iter_mut()
            .filter_map(|request| {
                request
                    .shares_id()
                    .map(|id| (generate_iris_shares(self.rng_mut()), id))
            })
            .collect();

        // Execute uploads in parallel.
        let aws_client = &self.aws_client;
        let tasks: Vec<_> = shares
            .iter()
            .map(|(shares, identifier)| async move {
                aws_client
                    .s3_upload_iris_shares(identifier, shares)
                    .await
                    .map_err(ClientError::AwsServiceError)
            })
            .collect();
        futures::future::try_join_all(tasks).await?;

        // Update state of requests.
        for item in batch.requests_mut() {
            item.set_status_data_uploaded();
        }

        Ok(())
    }
}
