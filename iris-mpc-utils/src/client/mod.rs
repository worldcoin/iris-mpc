use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;

use crate::aws::{AwsClient, AwsClientConfig};
use components::{
    RequestEnqueuer, RequestGenerator, RequestGeneratorParams, ResponseDequeuer, SharesGenerator,
    SharesUploader,
};
use options::IrisCodeSelectionStrategyOptions;
pub use options::{AwsOptions, ServiceClientOptions};
pub use typeset::ServiceClientError;
use typeset::{
    Initialize, IrisDescriptor, IrisPairDescriptor, ProcessRequestBatch, RequestBatch,
    RequestBatchKind, RequestBatchSize, UniquenessRequestDescriptor,
};

mod components;
mod options;
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
        opts: ServiceClientOptions,
        opts_aws: AwsOptions,
    ) -> Result<Self, ServiceClientError> {
        opts.validate()?;

        let aws_client = AwsClient::async_from(opts_aws).await;

        Ok(Self {
            shares_uploader: SharesUploader::new(
                aws_client.clone(),
                SharesGenerator::<R>::from(&opts),
            ),
            request_enqueuer: RequestEnqueuer::new(aws_client.clone()),
            request_generator: RequestGenerator::from(&opts),
            response_dequeuer: ResponseDequeuer::new(aws_client.clone()),
        })
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
impl AsyncFrom<AwsOptions> for AwsClient {
    async fn async_from(config: AwsOptions) -> Self {
        AwsClient::new(AwsClientConfig::async_from(config).await)
    }
}

#[async_from::async_trait]
impl AsyncFrom<AwsOptions> for AwsClientConfig {
    async fn async_from(config: AwsOptions) -> Self {
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

impl From<&ServiceClientOptions> for RequestGenerator {
    fn from(config: &ServiceClientOptions) -> Self {
        Self::new(RequestGeneratorParams::from(config))
    }
}

impl From<&ServiceClientOptions> for RequestGeneratorParams {
    fn from(opts: &ServiceClientOptions) -> Self {
        match opts.request_batch() {
            options::RequestBatchOptions::Simple {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => {
                tracing::info!("Parsing options::RequestBatchOptions::Simple");
                Self::Simple {
                    batch_count: *batch_count,
                    batch_size: RequestBatchSize::Static(*batch_size),
                    batch_kind: RequestBatchKind::from(batch_kind),
                    known_iris_serial_id: *known_iris_serial_id,
                }
            }
            options::RequestBatchOptions::Series {
                batches: opts_batches,
            } => {
                tracing::info!("Parsing options::RequestBatchOptions::Series");

                let batches: Vec<RequestBatch> = opts_batches
                    .iter()
                    .enumerate()
                    .map(|(batch_idx, opts_batch)| {
                        let mut batch = RequestBatch::new(batch_idx, vec![]);

                        for opts_request in opts_batch {
                            match opts_request.payload() {
                                options::RequestPayloadOptions::IdentityDeletion { parent } => {
                                    batch.push_new_identity_deletion(
                                        UniquenessRequestDescriptor::from(parent),
                                    );
                                }
                                options::RequestPayloadOptions::Reauthorisation {
                                    iris_pair,
                                    parent,
                                } => {
                                    batch.push_new_reauthorization(
                                        UniquenessRequestDescriptor::from(parent),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
                                }
                                options::RequestPayloadOptions::ResetCheck { iris_pair } => {
                                    batch.push_new_reset_check(Some(IrisPairDescriptor::from(
                                        iris_pair,
                                    )));
                                }
                                options::RequestPayloadOptions::ResetUpdate {
                                    iris_pair,
                                    parent,
                                } => {
                                    batch.push_new_reset_update(
                                        UniquenessRequestDescriptor::from(parent),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
                                }
                                options::RequestPayloadOptions::Uniqueness {
                                    iris_pair, ..
                                } => {
                                    batch.push_new_uniqueness(Some(IrisPairDescriptor::from(
                                        iris_pair,
                                    )));
                                }
                            }
                        }

                        batch
                    })
                    .collect();

                Self::Series(batches)
            }
        }
    }
}

impl<R: Rng + CryptoRng + SeedableRng + Send> From<&ServiceClientOptions> for SharesGenerator<R> {
    fn from(opts: &ServiceClientOptions) -> Self {
        match opts.shares_generator() {
            options::SharesGeneratorOptions::FromCompute { rng_seed } => {
                tracing::info!("Parsing options::SharesGeneratorOptions::FromCompute");
                SharesGenerator::<R>::new_compute(*rng_seed)
            }
            options::SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                rng_seed,
                selection_strategy,
            } => {
                tracing::info!("Parsing options::SharesGeneratorOptions::FromFile");
                SharesGenerator::new_file(
                    PathBuf::from(path_to_ndjson_file),
                    *rng_seed,
                    selection_strategy.as_ref().map(IrisSelection::from),
                )
            }
        }
    }
}

impl From<&options::UniquenessRequestDescriptorOptions> for UniquenessRequestDescriptor {
    fn from(opts: &options::UniquenessRequestDescriptorOptions) -> Self {
        match opts {
            options::UniquenessRequestDescriptorOptions::SerialId(serial_id) => {
                Self::IrisSerialId(*serial_id)
            }
            options::UniquenessRequestDescriptorOptions::Label(label) => {
                Self::new_label(label.clone())
            }
        }
    }
}

impl From<&options::IrisPairDescriptorOptions> for IrisPairDescriptor {
    fn from(opts: &options::IrisPairDescriptorOptions) -> Self {
        Self::new(
            IrisDescriptor::from(opts.left()),
            IrisDescriptor::from(opts.right()),
        )
    }
}

impl From<&options::IrisDescriptorOptions> for IrisDescriptor {
    fn from(opts: &options::IrisDescriptorOptions) -> Self {
        IrisDescriptor::new(opts.index(), opts.mutation())
    }
}

impl From<&IrisCodeSelectionStrategyOptions> for IrisSelection {
    fn from(opts: &IrisCodeSelectionStrategyOptions) -> Self {
        match opts {
            IrisCodeSelectionStrategyOptions::All => Self::All,
            IrisCodeSelectionStrategyOptions::Even => Self::Even,
            IrisCodeSelectionStrategyOptions::Odd => Self::Odd,
        }
    }
}
