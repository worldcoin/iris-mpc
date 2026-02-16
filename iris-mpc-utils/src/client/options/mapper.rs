use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use rand::{CryptoRng, Rng, SeedableRng};

use super::{
    super::ServiceClientOptions,
    types::{
        AwsOptions, RequestBatchOptions, RequestOptions, RequestPayloadOptions,
        SharesGeneratorOptions, UniquenessRequestDescriptorOptions,
    },
};
use crate::{
    aws::{AwsClient, AwsClientConfig},
    client::{
        components::{RequestGenerator, RequestGeneratorConfig, SharesGenerator},
        typeset::{BatchKind, RequestBatch, RequestBatchSet, UniquenessRequestDescriptor},
    },
};

#[async_from::async_trait]
impl AsyncFrom<AwsOptions> for AwsClient {
    async fn async_from(opts: AwsOptions) -> Self {
        AwsClient::new(AwsClientConfig::async_from(opts).await)
    }
}

#[async_from::async_trait]
impl AsyncFrom<AwsOptions> for AwsClientConfig {
    async fn async_from(opts: AwsOptions) -> Self {
        AwsClientConfig::new(
            opts.environment().to_owned(),
            opts.public_key_base_url().to_owned(),
            opts.s3_request_bucket_name().to_owned(),
            opts.sns_request_topic_arn().to_owned(),
            opts.sqs_long_poll_wait_time().to_owned(),
            opts.sqs_response_queue_url().to_owned(),
            opts.sqs_wait_time_seconds().to_owned(),
        )
        .await
    }
}

impl From<&Vec<Vec<RequestOptions>>> for RequestBatchSet {
    fn from(opts: &Vec<Vec<RequestOptions>>) -> Self {
        let batches: Vec<RequestBatch> = opts
            .iter()
            .enumerate()
            .map(|(batch_idx, opts_batch)| {
                let mut batch = RequestBatch::new(batch_idx, vec![]);
                for opts_request in opts_batch {
                    match opts_request.payload() {
                        RequestPayloadOptions::IdentityDeletion { parent } => {
                            batch.push_new_identity_deletion(
                                UniquenessRequestDescriptor::from(parent),
                                opts_request.label(),
                                parent.label(),
                            );
                        }
                        RequestPayloadOptions::Reauthorisation { iris_pair, parent } => {
                            batch.push_new_reauthorization(
                                UniquenessRequestDescriptor::from(parent),
                                Some(iris_pair.clone()),
                                opts_request.label(),
                                parent.label(),
                            );
                        }
                        RequestPayloadOptions::ResetCheck { iris_pair } => {
                            batch.push_new_reset_check(
                                Some(iris_pair.clone()),
                                opts_request.label(),
                            );
                        }
                        RequestPayloadOptions::ResetUpdate { iris_pair, parent } => {
                            batch.push_new_reset_update(
                                UniquenessRequestDescriptor::from(parent),
                                Some(iris_pair.clone()),
                                opts_request.label(),
                                parent.label(),
                            );
                        }
                        RequestPayloadOptions::Uniqueness { iris_pair, .. } => {
                            batch.push_new_uniqueness(
                                Some(iris_pair.clone()),
                                opts_request.label().clone(),
                            );
                        }
                    }
                }

                batch
            })
            .collect();

        RequestBatchSet::new(batches)
    }
}

impl From<&ServiceClientOptions> for RequestGenerator {
    fn from(opts: &ServiceClientOptions) -> Self {
        let mut config = RequestGeneratorConfig::from(opts);
        config.set_child_parent_descriptors_from_labels();

        Self::new(config)
    }
}

impl From<&ServiceClientOptions> for RequestGeneratorConfig {
    fn from(opts: &ServiceClientOptions) -> Self {
        match opts.request_batch() {
            RequestBatchOptions::Complex {
                batches: opts_batches,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Complex");
                Self::Complex(RequestBatchSet::from(opts_batches))
            }
            RequestBatchOptions::Simple {
                batch_count,
                batch_size,
                batch_kind,
                known_iris_serial_id,
            } => {
                tracing::info!("Parsing RequestBatchOptions::Simple");
                Self::Simple {
                    batch_count: *batch_count,
                    batch_size: *batch_size,
                    batch_kind: BatchKind::from_str(batch_kind)
                        .unwrap_or_else(|| panic!("Unsupported batch kind: {}", batch_kind)),
                    known_iris_serial_id: *known_iris_serial_id,
                }
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
                    *selection_strategy,
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
