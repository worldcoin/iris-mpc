use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;

use crate::{
    aws::{AwsClient, AwsClientConfig},
    client::config::IrisCodeSelectionStrategy,
};
use components::{
    RequestEnqueuer, RequestGenerator, RequestGeneratorParams, ResponseDequeuer, SharesGenerator,
    SharesGeneratorOptions, SharesUploader,
};
pub use config::{AwsConfiguration, ServiceClientConfiguration};
pub use typeset::ServiceClientError;
use typeset::{Initialize, ProcessRequestBatch, RequestBatchKind, RequestBatchSize};

mod components;
mod config;
mod typeset;

/// A utility for enqueuing system requests & correlating with system responses.
pub struct ServiceClient<R: Rng + CryptoRng + SeedableRng + Send> {
    // Component that enqueues system requests upon system ingress queues.
    request_enqueuer: RequestEnqueuer,

    // Component that generates system requests.
    request_generator: RequestGenerator,

    // Component that dequeues system responses from system egress queues.
    response_dequeuer: ResponseDequeuer,

    // Component that uploads iris shares to services prior to request processing.
    shares_uploader: SharesUploader<R>,
}

impl<R: Rng + CryptoRng + SeedableRng + Send> ServiceClient<R> {
    pub async fn new(
        client_config: ServiceClientConfiguration,
        aws_config: AwsConfiguration,
    ) -> Self {
        let aws_client = AwsClient::async_from(aws_config).await;

        Self {
            shares_uploader: SharesUploader::new(
                aws_client.clone(),
                SharesGenerator::<R>::from(&client_config),
            ),
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::from(&client_config),
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> ServiceClient<R> {
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
impl AsyncFrom<AwsConfiguration> for AwsClient {
    async fn async_from(config: AwsConfiguration) -> Self {
        AwsClient::new(AwsClientConfig::async_from(config).await)
    }
}

#[async_from::async_trait]
impl AsyncFrom<AwsConfiguration> for AwsClientConfig {
    async fn async_from(config: AwsConfiguration) -> Self {
        AwsClientConfig::new(
            config.environment().to_owned(),
            config.public_key_base_url().to_owned(),
            config.s3_request_bucket_name().to_owned(),
            config.sns_request_topic_arn().to_owned(),
            config.sqs_long_poll_wait_time().to_owned(),
            config.sqs_response_queue_url().to_owned(),
            config.sqs_wait_time_seconds().to_owned(),
        )
        .await
    }
}

impl From<&ServiceClientConfiguration> for RequestGenerator {
    fn from(config: &ServiceClientConfiguration) -> Self {
        Self::new(RequestGeneratorParams::from(config))
    }
}

impl From<&ServiceClientConfiguration> for RequestGeneratorParams {
    fn from(config: &ServiceClientConfiguration) -> Self {
        match config.request_batch() {
            config::RequestBatchConfiguration::Simple {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => {
                tracing::info!("Parsing config: Request batch set from simple kind");
                Self::Simple {
                    batch_count: *batch_count,
                    batch_size: RequestBatchSize::Static(*batch_size),
                    batch_kind: RequestBatchKind::from(batch_kind),
                    known_iris_serial_id: *known_iris_serial_id,
                }
            }
            config::RequestBatchConfiguration::KnownSet(request_batch_set) => {
                tracing::info!("Parsing config: Request batch set from known set");
                Self::KnownSet(request_batch_set.clone())
            }
        }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> From<&ServiceClientConfiguration>
    for SharesGenerator<R>
{
    fn from(config: &ServiceClientConfiguration) -> Self {
        Self::new(SharesGeneratorOptions::<R>::from(config))
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> From<&ServiceClientConfiguration>
    for SharesGeneratorOptions<R>
{
    fn from(config: &ServiceClientConfiguration) -> Self {
        match config.shares_generator() {
            config::SharesGeneratorConfiguration::FromFile {
                path_to_ndjson_file,
                rng_seed,
                selection_strategy,
            } => {
                tracing::info!("Parsing config: Shares generator from file");
                SharesGeneratorOptions::new_file(
                    PathBuf::from(path_to_ndjson_file),
                    *rng_seed,
                    selection_strategy.as_ref().map(IrisSelection::from),
                )
            }
            config::SharesGeneratorConfiguration::FromRng { rng_seed } => {
                tracing::info!("Parsing config: Shares generator from RNG");
                SharesGeneratorOptions::<R>::new_rng(*rng_seed)
            }
        }
    }
}

impl From<&IrisCodeSelectionStrategy> for IrisSelection {
    fn from(value: &IrisCodeSelectionStrategy) -> Self {
        match value {
            IrisCodeSelectionStrategy::All => Self::All,
            IrisCodeSelectionStrategy::Even => Self::Even,
            IrisCodeSelectionStrategy::Odd => Self::Odd,
        }
    }
}
