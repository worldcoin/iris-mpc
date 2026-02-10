use async_from::{self, AsyncFrom};

use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;

use crate::aws::AwsClientConfig;
use crate::client::options::{
    AwsOptions, IrisCodeSelectionStrategyOptions, IrisDescriptorOptions, IrisPairDescriptorOptions,
    UniquenessRequestDescriptorOptions,
};
use crate::client::typeset::{IrisDescriptor, IrisPairDescriptor, UniquenessRequestDescriptor};

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
