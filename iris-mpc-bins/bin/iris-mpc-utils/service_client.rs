use std::fmt;

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, SeedableRng};

use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;

use iris_mpc_utils::{
    aws::AwsClientConfig,
    client::{RequestBatchKind, RequestBatchSize, ServiceClient},
};

#[tokio::main]
pub async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let options = CliOptions::parse();
    tracing::info!("{}", options);

    tracing::info!("Initialising ...");
    let mut client = ServiceClient::async_from(options.clone()).await;
    match client.init(options.aws_public_key_base_url).await {
        Ok(()) => {
            client.exec().await.unwrap();
        }
        Err(e) => {
            tracing::error!("Initialisation failure: {}", e);
        }
    };

    Ok(())
}

#[derive(Debug, Parser, Clone)]
struct CliOptions {
    /// AWS: base URL for downloading node encryption public keys.
    #[clap(long)]
    aws_public_key_base_url: String,

    /// AWS: system request ingress queue URL.
    #[clap(long)]
    aws_s3_request_bucket_name: String,

    /// AWS: System SNS request queue topic.
    #[clap(long)]
    aws_sns_request_topic_arn: String,

    /// AWS: polling interval between AWS SQS interactions.
    #[clap(long, default_value = "10")]
    aws_sqs_long_poll_wait_time: usize,

    /// AWS: system response egress queue URL.
    #[clap(long)]
    aws_sqs_response_queue_url: String,

    #[clap(long)]
    environment: String,

    /// Number of request batches to process.
    #[clap(long, default_value = "5")]
    request_batch_count: usize,

    /// Maximum size of each request batch.
    #[clap(long, default_value = "10")]
    request_batch_size: usize,

    /// A random number generator seed for upstream entropy.
    #[clap(long)]
    rng_seed: Option<u64>,
}

impl CliOptions {
    fn request_batch_kind(&self) -> RequestBatchKind {
        // Currently defaults to sets of uniqueness requests.
        // TODO: parse from env | command line | file.
        RequestBatchKind::Simple(UNIQUENESS_MESSAGE_TYPE)
    }

    fn request_batch_size(&self) -> RequestBatchSize {
        // Currently defaults to static batches.
        RequestBatchSize::Static(self.request_batch_size)
    }

    #[allow(dead_code)]
    fn rng_seed(&self) -> StdRng {
        if self.rng_seed.is_some() {
            StdRng::seed_from_u64(self.rng_seed.unwrap())
        } else {
            StdRng::from_entropy()
        }
    }
}

impl fmt::Display for CliOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
Iris-MPC Service Client Options:
    aws_public_key_base_url
        {}
    aws_s3_request_bucket_name
        {}
    aws_sns_request_topic_arn
        {}
    aws_sqs_long_poll_wait_time
        {}
    aws_sqs_response_queue_url
        {}
    environment
        {}
    request_batch_count
        {}
    request_batch_kind
        {:?}
    request_batch_size
        {:?}
    rng_seed
        {:?}
                ",
            self.aws_public_key_base_url,
            self.aws_s3_request_bucket_name,
            self.aws_sns_request_topic_arn,
            self.aws_sqs_long_poll_wait_time,
            self.aws_sqs_response_queue_url,
            self.environment,
            self.request_batch_count,
            self.request_batch_kind(),
            self.request_batch_size(),
            self.rng_seed,
        )
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for ServiceClient<StdRng> {
    async fn async_from(options: CliOptions) -> Self {
        ServiceClient::new(
            AwsClientConfig::async_from(options.clone()).await,
            options.request_batch_count,
            options.request_batch_kind(),
            options.request_batch_size(),
            options.rng_seed(),
        )
        .await
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for AwsClientConfig {
    async fn async_from(options: CliOptions) -> Self {
        AwsClientConfig::new(
            options.environment,
            options.aws_s3_request_bucket_name,
            options.aws_sns_request_topic_arn,
            options.aws_sqs_long_poll_wait_time,
            options.aws_sqs_response_queue_url,
        )
        .await
    }
}
