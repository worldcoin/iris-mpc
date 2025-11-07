use std::path::PathBuf;

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, SeedableRng};

use iris_mpc_common::helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE;
use iris_mpc_utils::{
    aws::{
        download_net_encryption_public_keys, AwsClient, AwsClientConfig, NetAwsClient,
        NetAwsClientConfig, AWS_PUBLIC_KEY_BASE_URL, AWS_REQUESTS_BUCKET_NAME,
        AWS_REQUESTS_TOPIC_ARN, AWS_RESPONSE_QUEUE_URL,
    },
    client::{self, AwsRequestDispatcher, Client, RequestGenerator},
    constants::NODE_CONFIG_KIND_MAIN,
    fsys,
    types::{NetEncryptionPublicKeys, NetNodeConfig},
};

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = CliOptions::parse();
    println!("Instantiated options: {:?}", &options);

    let mut client = Client::async_from(options.clone()).await;
    println!("Instantiated client");

    println!("Running client ...");
    client.run().await;

    Ok(())
}

#[derive(Debug, Parser, Clone)]
struct CliOptions {
    /// AWS: base URL for downloading node encryption public keys.
    #[clap(long)]
    aws_public_key_base_url: Option<String>,

    /// AWS: system request ingress queue topic.
    #[clap(long)]
    aws_request_topic_arn: Option<String>,

    /// AWS: system request ingress queue URL.
    #[clap(long)]
    aws_requests_bucket_name: Option<String>,

    /// AWS: system response egress queue URL.
    #[clap(long)]
    aws_response_queue_url: Option<String>,

    /// Maximum size of each request batch.
    #[clap(long, default_value = "500")]
    batch_size: usize,

    /// Number of request batches to process.
    #[clap(long, default_value = "1")]
    n_batches: usize,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_0_config: Option<String>,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_1_config: Option<String>,

    /// Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_2_config: Option<String>,

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

    fn aws_requests_bucket_name(&self) -> String {
        self.aws_requests_bucket_name
            .as_deref()
            .unwrap_or(AWS_REQUESTS_BUCKET_NAME)
            .to_string()
    }

    fn aws_request_topic_arn(&self) -> String {
        self.aws_request_topic_arn
            .as_deref()
            .unwrap_or(AWS_REQUESTS_TOPIC_ARN)
            .to_string()
    }

    fn aws_response_queue_url(&self) -> String {
        self.aws_response_queue_url
            .as_deref()
            .unwrap_or(AWS_RESPONSE_QUEUE_URL)
            .to_string()
    }

    fn batch_size(&self) -> &usize {
        &self.batch_size
    }

    fn n_batches(&self) -> &usize {
        &self.n_batches
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
impl AsyncFrom<CliOptions> for Client<AwsRequestDispatcher, RequestGenerator> {
    async fn async_from(options: CliOptions) -> Self {
        // let _ = NetEncryptionPublicKeys::async_from(options.clone()).await;
        Client::new(
            AwsRequestDispatcher::async_from(options.clone()).await,
            RequestGenerator::from(&options),
        )
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetEncryptionPublicKeys {
    async fn async_from(options: CliOptions) -> Self {
        download_net_encryption_public_keys(&options.aws_public_key_base_url())
            .await
            .unwrap()
    }
}

impl From<CliOptions> for NetNodeConfig {
    fn from(options: CliOptions) -> Self {
        [
            fsys::read_node_config(&options.path_to_node_0_config()).unwrap(),
            fsys::read_node_config(options.path_to_node_1_config().as_path()).unwrap(),
            fsys::read_node_config(options.path_to_node_2_config().as_path()).unwrap(),
        ]
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetAwsClient {
    async fn async_from(options: CliOptions) -> Self {
        NetAwsClientConfig::async_from(options)
            .await
            .map(|x| AwsClient::new(x))
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetAwsClientConfig {
    async fn async_from(options: CliOptions) -> Self {
        let node_configs = NetNodeConfig::from(options.clone());

        [
            AwsClientConfig::new(
                node_configs[0].to_owned(),
                options.aws_public_key_base_url(),
                options.aws_request_topic_arn(),
                options.aws_requests_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
            AwsClientConfig::new(
                node_configs[1].to_owned(),
                options.aws_public_key_base_url(),
                options.aws_request_topic_arn(),
                options.aws_requests_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
            AwsClientConfig::new(
                node_configs[2].to_owned(),
                options.aws_public_key_base_url(),
                options.aws_request_topic_arn(),
                options.aws_requests_bucket_name(),
                options.aws_response_queue_url(),
            )
            .await,
        ]
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for AwsRequestDispatcher {
    async fn async_from(options: CliOptions) -> Self {
        Self::new(NetAwsClient::async_from(options).await)
    }
}

impl From<&CliOptions> for client::RequestGenerator {
    fn from(options: &CliOptions) -> Self {
        Self::new(
            client::RequestBatchKind::Simple(UNIQUENESS_MESSAGE_TYPE),
            client::RequestBatchSize::Static(*options.batch_size()),
            *options.n_batches(),
        )
    }
}
