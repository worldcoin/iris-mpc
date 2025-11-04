use std::path::Path;

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, SeedableRng};

use iris_mpc_common::{
    config::Config as NodeConfig, helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_utils::{
    constants::NODE_CONFIG_KIND_MAIN,
    service_client as client,
    state::{
        aws::{
            download_net_encryption_public_keys, NodeAwsClient as NodeServiceClient,
            NodeAwsConfig as NodeServiceConfig,
        },
        fsys::{
            local::get_path_to_node_config as get_path_to_local_node_config,
            reader::read_node_config,
        },
    },
    types::{NetEncryptionPublicKeys, NetNodeConfig, NetServiceClients, NetServiceConfig},
};

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = CliOptions::parse();
    println!("Instantiated options: {:?}", &options);

    let _ = NetEncryptionPublicKeys::async_from(options.clone()).await;

    let mut client = client::Client::new(
        client::AwsRequestDispatcher::async_from(options.clone()).await,
        client::RequestGenerator::from(&options),
    );
    println!("Instantiated client");

    println!("Running client ...");
    client.run().await;

    Ok(())
}

#[derive(Debug, Parser, Clone)]
struct CliOptions {
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

    /// Base URL to use when downloading node encryption public keys.
    #[clap(long, default_value = "http://localhost:4566/wf-dev-public-keys")]
    public_key_base_url: String,

    /// A random number generator seed for upstream entropy.
    #[clap(long)]
    rng_seed: Option<u64>,
}

impl CliOptions {
    fn batch_size(&self) -> &usize {
        &self.batch_size
    }

    fn n_batches(&self) -> &usize {
        &self.n_batches
    }

    fn path_to_node_0_config(&self) -> &Option<String> {
        &self.path_to_node_0_config
    }

    fn path_to_node_1_config(&self) -> &Option<String> {
        &self.path_to_node_1_config
    }

    fn path_to_node_2_config(&self) -> &Option<String> {
        &self.path_to_node_2_config
    }

    fn public_key_base_url(&self) -> &String {
        &self.public_key_base_url
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

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetEncryptionPublicKeys {
    async fn async_from(options: CliOptions) -> Self {
        download_net_encryption_public_keys(options.public_key_base_url())
            .await
            .unwrap()
    }
}

impl From<CliOptions> for NetNodeConfig {
    fn from(options: CliOptions) -> Self {
        [
            read_config_of_node(0, options.path_to_node_0_config()),
            read_config_of_node(1, options.path_to_node_1_config()),
            read_config_of_node(2, options.path_to_node_2_config()),
        ]
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetServiceClients {
    async fn async_from(options: CliOptions) -> Self {
        NetServiceConfig::async_from(options)
            .await
            .map(|x| NodeServiceClient::new(x))
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for NetServiceConfig {
    async fn async_from(options: CliOptions) -> Self {
        let node_configs = NetNodeConfig::from(options);
        [
            NodeServiceConfig::new(node_configs[0].to_owned()).await,
            NodeServiceConfig::new(node_configs[1].to_owned()).await,
            NodeServiceConfig::new(node_configs[2].to_owned()).await,
        ]
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for client::AwsRequestDispatcher {
    async fn async_from(options: CliOptions) -> Self {
        Self::new(NetServiceClients::async_from(options).await)
    }
}

impl From<&CliOptions> for client::RequestGenerator {
    fn from(options: &CliOptions) -> Self {
        Self::new(
            client::BatchKind::Simple(UNIQUENESS_MESSAGE_TYPE),
            client::BatchSize::Static(*options.batch_size()),
            *options.n_batches(),
        )
    }
}

fn read_config_of_node(party_idx: usize, path: &Option<String>) -> NodeConfig {
    let path_to_config = match path {
        Some(path) => path.clone(),
        None => get_path_to_local_node_config(NODE_CONFIG_KIND_MAIN, 0, &party_idx)
            .to_string_lossy()
            .to_string(),
    };

    read_node_config(Path::new(&path_to_config)).unwrap()
}
