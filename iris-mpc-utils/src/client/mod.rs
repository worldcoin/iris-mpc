use rand::{CryptoRng, Rng};

use crate::aws::{AwsClient, AwsClientConfig};

pub use components::RequestGeneratorParams;
use components::{RequestEnqueuer, RequestGenerator, ResponseDequeuer, SharesUploader};
pub use typeset::{
    Initialize, ProcessRequestBatch, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
    ServiceClientConfig, ServiceClientError,
};

mod components;
mod typeset;

/// A utility for enqueuing system requests & correlating with system responses.
#[derive(Debug)]
pub struct ServiceClient<R: Rng + CryptoRng + Send> {
    // Component that enqueues system requests upon system ingress queues.
    request_enqueuer: RequestEnqueuer,

    // Component that generates system requests.
    request_generator: RequestGenerator,

    // Component that dequeues system responses from system egress queues.
    response_dequeuer: ResponseDequeuer,

    // Component that uploads iris shares to services prior to request processing.
    shares_uploader: SharesUploader<R>,
}

impl<R: Rng + CryptoRng + Send> ServiceClient<R> {
    pub async fn new(
        aws_client_config: AwsClientConfig,
        config: ServiceClientConfig,
        rng_seed: R,
    ) -> Self {
        let aws_client = AwsClient::new(aws_client_config);

        Self {
            shares_uploader: SharesUploader::new(aws_client.clone(), rng_seed),
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::new(config),
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        }
    }

    pub async fn exec(&mut self) -> Result<(), ServiceClientError> {
        while let Some(mut batch) = self.request_generator.next().await.unwrap() {
            println!("------------------------------------------------------------------------");
            println!(
                "Batch {}: size={}",
                batch.batch_idx(),
                batch.requests().len()
            );
            println!("------------------------------------------------------------------------");
            self.shares_uploader.process_batch(&mut batch).await?;
            while batch.is_enqueueable() {
                self.request_enqueuer.process_batch(&mut batch).await?;
                self.response_dequeuer.process_batch(&mut batch).await?;
            }
        }

        Ok(())
    }

    pub async fn init(&mut self) -> Result<(), ServiceClientError> {
        for initializer in [self.shares_uploader.init(), self.response_dequeuer.init()] {
            initializer.await.map_err(|e| {
                tracing::error!("Service client: component initialisation failed: {}", e);
                ServiceClientError::InitialisationError(e.to_string())
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::{AwsClientConfig, ServiceClient, ServiceClientConfig};

    impl ServiceClient<StdRng> {
        async fn new_1() -> Self {
            ServiceClient::<StdRng>::new(
                AwsClientConfig::new_1().await,
                ServiceClientConfig::new_1(),
                StdRng::seed_from_u64(42),
            )
            .await
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let _ = ServiceClient::new_1().await;
    }
}
