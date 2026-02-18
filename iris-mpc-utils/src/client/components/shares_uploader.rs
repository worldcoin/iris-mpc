use async_trait::async_trait;
use futures;
use rand::{CryptoRng, Rng, SeedableRng};

use super::{
    super::typeset::{
        Initialize, ProcessRequestBatch, RequestBatch, RequestStatus, ServiceClientError,
    },
    SharesGenerator,
};
use crate::aws::AwsClient;

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

        // Upload shares for active requests.
        for request in batch.requests() {
            if let Some((identifier, iris_pair)) = request.get_shares_info() {
                shares.push((
                    identifier,
                    self.shares_generator.generate(iris_pair.as_ref()),
                ));
            }
        }

        // Upload shares for pending items that have an iris_pair
        // (Complex mode cross-batch children whose shares must be ready before activation).
        for item in batch.pending() {
            let (op_id, iris_pair) = item.shares_info();
            if let Some(iris_pair) = iris_pair {
                shares.push((op_id, self.shares_generator.generate(Some(iris_pair))));
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

        // Update state of active requests.
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
        #[allow(dead_code)]
        pub async fn new_1() -> Self {
            Self::new(AwsClient::new_1().await, SharesGenerator::new_compute_1())
        }
    }

    impl<R: Rng + CryptoRng + SeedableRng + Send> SharesUploader<R> {
        #[allow(dead_code)]
        pub async fn new_2() -> Self {
            Self::new(AwsClient::new_1().await, SharesGenerator::<R>::new_file_1())
        }
    }
}
