use async_trait::async_trait;
use futures;
use rand::{CryptoRng, Rng, SeedableRng};

use super::{
    super::typeset::{
        Initialize, ProcessRequestBatch, RequestBatch, RequestStatus, ServiceClientError,
    },
    SharesGenerator,
};
use crate::{aws::AwsClient, client::typeset::Request};

/// A component responsible for uploading Iris shares to AWS services
/// in advance of system request processing.
pub(crate) struct SharesUploader<R: Rng + CryptoRng + SeedableRng + Send> {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,

    /// Iris shares provider.
    shares_generator: SharesGenerator<R>,
}

impl<R: Rng + CryptoRng + SeedableRng + Send> SharesUploader<R> {
    pub fn new(aws_client: AwsClient, shares_generator: SharesGenerator<R>) -> Self {
        Self {
            aws_client,
            shares_generator,
        }
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + SeedableRng + Send> Initialize for SharesUploader<R> {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        self.shares_generator.init().await?;
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ServiceClientError::AwsServiceError)
    }
}

#[async_trait]
impl<R: Rng + CryptoRng + SeedableRng + Send> ProcessRequestBatch for SharesUploader<R> {
    async fn process_batch(&mut self, batch: &mut RequestBatch) -> Result<(), ServiceClientError> {
        // Set shares to be uploaded.
        let mut shares: Vec<_> = Vec::new();
        for request in batch.requests_mut().iter_mut() {
            if let Some(identifier) = match request {
                Request::IdentityDeletion { .. } => None,
                Request::Reauthorization { reauth_id, .. } => Some(reauth_id),
                Request::ResetCheck { reset_id, .. } => Some(reset_id),
                Request::ResetUpdate { reset_id, .. } => Some(reset_id),
                Request::Uniqueness { signup_id, .. } => Some(signup_id),
            } {
                shares.push((identifier, self.shares_generator.generate()));
            }
        }

        // Execute uploads in parallel.
        let aws_client = &self.aws_client;
        let tasks: Vec<_> = shares
            .iter()
            .map(|(identifier, shares)| async move {
                aws_client
                    .s3_upload_iris_shares(identifier, shares)
                    .await
                    .map_err(ServiceClientError::AwsServiceError)
            })
            .collect();
        futures::future::try_join_all(tasks).await?;

        // Update state of requests.
        batch.set_request_status(RequestStatus::SharesUploaded);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{super::shares_generator::SharesGenerator, SharesUploader};
    use crate::aws::AwsClient;
    use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

    impl SharesUploader<StdRng> {
        pub async fn new_1() -> Self {
            Self::new(AwsClient::new_1().await, SharesGenerator::new_1())
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesUploader<R> {
        pub async fn new_2() -> Self {
            Self::new(AwsClient::new_1().await, SharesGenerator::<R>::new_2())
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = SharesUploader::new_1().await;
    }

    #[tokio::test]
    async fn test_new_2() {
        let _ = SharesUploader::<StdRng>::new_2().await;
    }
}
