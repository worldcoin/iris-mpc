use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, SeedableRng};

use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;
use iris_mpc_utils::{
    aws::{
        NetAwsClientConfig, NodeAwsClientConfig, AWS_PUBLIC_KEY_BASE_URL, AWS_REQUEST_BUCKET_NAME,
        AWS_REQUEST_TOPIC_ARN, AWS_RESPONSE_QUEUE_URL,
    },
    client::{Client, RequestBatchKind, RequestBatchSize},
    constants::NODE_CONFIG_KIND_MAIN,
    fsys,
    types::NetNodeConfig,
};

#[tokio::main]
pub async fn main() -> Result<()> {
    println!("Options: instantiating ...");
    let options = CliOptions::parse();

    println!("Client: instantiating ...");
    let mut client = Client::async_from(options.clone()).await;

    println!("Client: running ...");
    client.run().await;

    Ok(())
}

#[derive(Debug, Parser, Clone)]
struct CliOptions {
    /// AWS: base URL for downloading node encryption public keys.
    #[clap(long)]
    aws_public_key_base_url: Option<String>,

    /// AWS: system request ingress queue URL.
    #[clap(long)]
    aws_request_bucket_name: Option<String>,

    /// AWS: system request ingress queue topic.
    #[clap(long)]
    aws_request_topic_arn: Option<String>,

    /// AWS: system response egress queue URL.
    #[clap(long)]
    aws_response_queue_url: Option<String>,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_0_config: Option<String>,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_1_config: Option<String>,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_2_config: Option<String>,

    /// Number of request batches to process.
    #[clap(long, default_value = "1")]
    request_batch_count: usize,

    /// Maximum size of each request batch.
    #[clap(long, default_value = "500")]
    request_batch_size: usize,

    /// A random number generator seed for upstream entropy.
    #[clap(long)]
    rng_seed: Option<u64>,
}

impl CliOptions {
    fn aws_public_key_base_url(&self) -> String {
        self.aws_public_key_base_url
            .as_deref()
            .unwrap_or(AWS_PUBLIC_KEY_BASE_URL)
            .to_string()
    }

    fn aws_request_bucket_name(&self) -> String {
        self.aws_request_bucket_name
            .as_deref()
            .unwrap_or(AWS_REQUEST_BUCKET_NAME)
            .to_string()
    }

    fn aws_request_topic_arn(&self) -> String {
        self.aws_request_topic_arn
            .as_deref()
            .unwrap_or(AWS_REQUEST_TOPIC_ARN)
            .to_string()
    }

    fn aws_response_queue_url(&self) -> String {
        self.aws_response_queue_url
            .as_deref()
            .unwrap_or(AWS_RESPONSE_QUEUE_URL)
            .to_string()
    }

    fn path_to_node_0_config(&self) -> PathBuf {
        Self::get_path_to_node_config(&self.path_to_node_0_config, 0)
    }

    fn path_to_node_1_config(&self) -> PathBuf {
        Self::get_path_to_node_config(&self.path_to_node_1_config, 1)
    }

    fn path_to_node_2_config(&self) -> PathBuf {
        Self::get_path_to_node_config(&self.path_to_node_2_config, 2)
    }

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

    /// Returns path to a node's config file - read locally if necessary.
    fn get_path_to_node_config(path: &Option<String>, party_idx: usize) -> PathBuf {
        path.clone()
            .unwrap_or_else(|| {
                fsys::local::get_path_to_node_config(NODE_CONFIG_KIND_MAIN, 0, &party_idx)
                    .to_string_lossy()
                    .to_string()
            })
            .into()
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for Client {
    async fn async_from(options: CliOptions) -> Self {
        Client::new(
            NetAwsClientConfig::async_from(options.clone()).await,
            options.aws_public_key_base_url(),
            options.request_batch_count,
            options.request_batch_kind(),
            options.request_batch_size(),
            options.rng_seed(),
        )
        .await
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetAwsClientConfig {
    async fn async_from(options: CliOptions) -> Self {
        let node_configs = NetNodeConfig::from(options.clone());

        [
            NodeAwsClientConfig::new(
                node_configs[0].to_owned(),
                options.aws_request_topic_arn(),
                options.aws_request_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
            NodeAwsClientConfig::new(
                node_configs[1].to_owned(),
                options.aws_request_topic_arn(),
                options.aws_request_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
            NodeAwsClientConfig::new(
                node_configs[2].to_owned(),
                options.aws_request_topic_arn(),
                options.aws_request_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
        ]
    }
}

impl From<CliOptions> for NetNodeConfig {
    fn from(options: CliOptions) -> Self {
        [
            fsys::read_node_config(&options.path_to_node_0_config()).unwrap(),
            fsys::read_node_config(&options.path_to_node_1_config()).unwrap(),
            fsys::read_node_config(&options.path_to_node_2_config()).unwrap(),
        ]
    }
}
