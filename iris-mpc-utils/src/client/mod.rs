use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng};

use crate::aws::{AwsClient, AwsClientConfig};

pub use components::RequestGeneratorParams;
use components::{RequestEnqueuer, RequestGenerator, ResponseDequeuer, SharesUploader};
pub use typeset::{
    config::{self, ServiceClientConfiguration},
    Initialize, ProcessRequestBatch, Request, RequestBatch, RequestBatchKind, RequestBatchSize,
    ServiceClientError,
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
    pub async fn new(config: config::ServiceClientConfiguration, rng_seed: R) -> Self {
        let aws_client = AwsClient::async_from(config.clone()).await;

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
                "Processing Batch {}: size={}",
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

#[async_from::async_trait]
impl AsyncFrom<ServiceClientConfiguration> for AwsClient {
    async fn async_from(config: ServiceClientConfiguration) -> Self {
        AwsClient::new(AwsClientConfig::async_from(config).await)
    }
}

#[async_from::async_trait]
impl AsyncFrom<ServiceClientConfiguration> for AwsClientConfig {
    async fn async_from(config: ServiceClientConfiguration) -> Self {
        AwsClientConfig::new(
            config.aws().environment().to_owned(),
            config.aws().public_key_base_url().to_owned(),
            config.aws().s3_request_bucket_name().to_owned(),
            config.aws().sns_request_topic_arn().to_owned(),
            config.aws().sqs_long_poll_wait_time().to_owned(),
            config.aws().sqs_response_queue_url().to_owned(),
            config.aws().sqs_wait_time_seconds().to_owned(),
        )
        .await
    }
}
