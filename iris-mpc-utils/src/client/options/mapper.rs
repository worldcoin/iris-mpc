use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;

use crate::{
    aws::{AwsClient, AwsClientConfig},
    client::{
        components::{RequestGenerator, RequestGeneratorParams, SharesGenerator},
        options::{
            AwsOptions, IrisCodeSelectionStrategyOptions, IrisDescriptorOptions,
            IrisPairDescriptorOptions, RequestBatchOptions, RequestPayloadOptions,
            ServiceClientOptions, SharesGeneratorOptions, UniquenessRequestDescriptorOptions,
        },
        typeset::{
            IrisDescriptor, IrisPairDescriptor, RequestBatch, RequestBatchKind, RequestBatchSize,
            UniquenessRequestDescriptor,
        },
    },
};

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

impl From<&IrisPairDescriptorOptions> for IrisPairDescriptor {
    fn from(opts: &IrisPairDescriptorOptions) -> Self {
        Self::new(
            IrisDescriptor::from(opts.left()),
            IrisDescriptor::from(opts.right()),
        )
    }
}

impl From<&IrisDescriptorOptions> for IrisDescriptor {
    fn from(opts: &IrisDescriptorOptions) -> Self {
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

impl From<&ServiceClientOptions> for RequestGenerator {
    fn from(config: &ServiceClientOptions) -> Self {
        Self::new(RequestGeneratorParams::from(config))
    }
}

impl From<&ServiceClientOptions> for RequestGeneratorParams {
    fn from(opts: &ServiceClientOptions) -> Self {
        match opts.request_batch() {
            RequestBatchOptions::Simple {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Simple");
                Self::Simple {
                    batch_count: *batch_count,
                    batch_size: RequestBatchSize::Static(*batch_size),
                    batch_kind: RequestBatchKind::from(batch_kind),
                    known_iris_serial_id: *known_iris_serial_id,
                }
            }
            RequestBatchOptions::Series {
                batches: opts_batches,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Series");

                let batches: Vec<RequestBatch> = opts_batches
                    .iter()
                    .enumerate()
                    .map(|(batch_idx, opts_batch)| {
                        let mut batch = RequestBatch::new(batch_idx, vec![]);
                        for opts_request in opts_batch {
                            match opts_request.payload() {
                                RequestPayloadOptions::IdentityDeletion { parent } => {
                                    batch.push_new_identity_deletion(
                                        opts_request.label().clone(),
                                        UniquenessRequestDescriptor::from(parent),
                                    );
                                }
                                RequestPayloadOptions::Reauthorisation { iris_pair, parent } => {
                                    batch.push_new_reauthorization(
                                        opts_request.label().clone(),
                                        UniquenessRequestDescriptor::from(parent),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
                                }
                                RequestPayloadOptions::ResetCheck { iris_pair } => {
                                    batch.push_new_reset_check(
                                        opts_request.label().clone(),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
                                }
                                RequestPayloadOptions::ResetUpdate { iris_pair, parent } => {
                                    batch.push_new_reset_update(
                                        opts_request.label().clone(),
                                        UniquenessRequestDescriptor::from(parent),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
                                }
                                RequestPayloadOptions::Uniqueness { iris_pair, .. } => {
                                    batch.push_new_uniqueness(
                                        opts_request.label().clone(),
                                        Some(IrisPairDescriptor::from(iris_pair)),
                                    );
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
            SharesGeneratorOptions::FromCompute { rng_seed } => {
                tracing::info!("Parsing SharesGeneratorOptions::FromCompute");
                SharesGenerator::<R>::new_compute(*rng_seed)
            }
            SharesGeneratorOptions::FromFile {
                path_to_ndjson_file,
                rng_seed,
                selection_strategy,
            } => {
                tracing::info!("Parsing SharesGeneratorOptions::FromFile");
                SharesGenerator::new_file(
                    PathBuf::from(path_to_ndjson_file),
                    *rng_seed,
                    selection_strategy.as_ref().map(IrisSelection::from),
                )
            }
        }
    }
}

impl From<&UniquenessRequestDescriptorOptions> for UniquenessRequestDescriptor {
    fn from(opts: &UniquenessRequestDescriptorOptions) -> Self {
        match opts {
            UniquenessRequestDescriptorOptions::SerialId(serial_id) => {
                Self::IrisSerialId(*serial_id)
            }
            UniquenessRequestDescriptorOptions::Label(label) => Self::Label(label.clone()),
        }
    }
}
