use std::path::PathBuf;

use async_trait::async_trait;
use futures;
use rand::{CryptoRng, Rng};

use super::super::typeset::{
    GenerateShares, Initialize, ProcessRequestBatch, RequestBatch, RequestStatus,
    ServiceClientError,
};
use super::shares_generator::{SharesGeneratorFromCompute, SharesGeneratorFromFile};
use crate::{aws::AwsClient, client::typeset::Request};

/// A component responsible for uploading Iris shares to AWS services
/// in advance of system request processing.
#[derive(Debug)]
pub(crate) struct SharesUploader<GS: GenerateShares + Send> {
    /// A client for interacting with system AWS services.
    aws_client: AwsClient,

    /// Iris shares provider.
    shares_generator: GS,
}

impl<GS: GenerateShares + Send> SharesUploader<GS> {
    fn shares_provider_mut(&mut self) -> &mut GS {
        &mut self.shares_generator
    }

    pub fn new(aws_client: AwsClient, shares_provider: GS) -> Self {
        Self {
            aws_client,
            shares_generator: shares_provider,
        }
    }
}

impl<R: Rng + CryptoRng + Send> SharesUploader<SharesGeneratorFromCompute<R>> {
    pub fn new_from_compute(aws_client: AwsClient, rng: R) -> Self {
        Self::new(aws_client, SharesGeneratorFromCompute::new(rng))
    }
}

impl SharesUploader<SharesGeneratorFromFile> {
    pub fn new_from_file(aws_client: AwsClient, path_to_ndjson_file: PathBuf) -> Self {
        Self::new(
            aws_client,
            SharesGeneratorFromFile::new(path_to_ndjson_file),
        )
    }
}

#[async_trait]
impl<GS: GenerateShares + Send> Initialize for SharesUploader<GS> {
    async fn init(&mut self) -> Result<(), ServiceClientError> {
        self.aws_client
            .set_public_keyset()
            .await
            .map_err(ServiceClientError::AwsServiceError)
    }
}

#[async_trait]
impl<GS: GenerateShares + Send> ProcessRequestBatch for SharesUploader<GS> {
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
                shares.push((identifier, self.shares_generator.generate().await));
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
    use super::{
        super::shares_generator::{SharesGeneratorFromCompute, SharesGeneratorFromFile},
        SharesUploader,
    };
    use crate::aws::AwsClient;
    use rand::rngs::StdRng;

    impl SharesUploader<SharesGeneratorFromCompute<StdRng>> {
        pub async fn new_1() -> Self {
            Self::new(
                AwsClient::new_1().await,
                SharesGeneratorFromCompute::new_1(),
            )
        }
    }

    impl SharesUploader<SharesGeneratorFromFile> {
        pub async fn new_2() -> Self {
            Self::new(AwsClient::new_1().await, SharesGeneratorFromFile::new_1())
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = SharesUploader::new_1();
    }

    #[tokio::test]
    async fn test_new_2() {
        let _ = SharesUploader::new_2();
    }
}
