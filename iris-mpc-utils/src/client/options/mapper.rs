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
            IrisPairDescriptorOptions, ServiceClientOptions, SharesGeneratorOptions,
            UniquenessRequestDescriptorOptions,
        },
        typeset::{IrisDescriptor, IrisPairDescriptor, UniquenessRequestDescriptor},
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
