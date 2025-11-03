use std::path::Path;

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;

use iris_mpc_common::{
    config::Config as NodeConfig, helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_utils::{
    constants::NODE_CONFIG_KIND_MAIN,
    state::{
        aws::{NodeAwsClient as NodeServiceClient, NodeAwsConfig as NodeServiceConfig},
        fsys::{
            local::get_path_to_node_config as get_path_to_local_node_config,
            reader::read_node_config,
        },
    },
    types::{NetNodeConfig, NetServiceClients, NetServiceConfig},
};

mod client;
mod requests;
mod responses;

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = CliOptions::parse();
    println!("Instantiated options: {:?}", &options);

    let mut client = client::Client::new(
        requests::AwsDispatcher::async_from(options.clone()).await,
        requests::Generator::from(&options),
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

    // Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_0_config: Option<String>,

    // Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_1_config: Option<String>,

    // Path to SMPC node-0 configuration file.
    #[clap(long)]
    path_to_node_2_config: Option<String>,
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
}

impl From<&CliOptions> for NetNodeConfig {
    fn from(options: &CliOptions) -> Self {
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
        [
            NodeServiceConfig::new(read_config_of_node(0, options.path_to_node_0_config())).await,
            NodeServiceConfig::new(read_config_of_node(1, options.path_to_node_1_config())).await,
            NodeServiceConfig::new(read_config_of_node(2, options.path_to_node_2_config())).await,
        ]
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for requests::AwsDispatcher {
    async fn async_from(options: CliOptions) -> Self {
        Self::new(NetServiceClients::async_from(options).await)
    }
}

impl From<&CliOptions> for requests::Factory {
    fn from(_: &CliOptions) -> Self {
        Self::new(requests::BatchKind::Simple(UNIQUENESS_MESSAGE_TYPE))
    }
}

impl From<&CliOptions> for requests::Generator<requests::Factory> {
    fn from(options: &CliOptions) -> Self {
        Self::new(
            requests::BatchSize::Static(*options.batch_size()),
            *options.n_batches(),
            requests::Factory::from(options),
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
