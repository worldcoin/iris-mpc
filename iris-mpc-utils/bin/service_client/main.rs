use std::path::Path;

use clap::Parser;
use eyre::Result;

use iris_mpc_common::{
    config::Config as NodeConfig, helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_utils::{
    constants::NODE_CONFIG_KIND_MAIN,
    state::fsys::{
        local::get_path_to_node_config as get_path_to_local_node_config, reader::read_node_config,
    },
    types::NetConfig,
};

mod requests;
mod responses;

use requests::{BatchDispatcher, BatchIterator};

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = CliOptions::parse();
    println!("Running with options: {:?}", &options,);

    let requests_dispatcher = requests::Dispatcher::from(&options);
    let mut requests_generator = requests::BatchGenerator::from(&options);

    while let Some(request_batch) = requests_generator.next_batch().await {
        println!("Request batch generated: {:?}", request_batch);
        requests_dispatcher.dispatch_batch(request_batch).await;
    }

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

    /// Returns network config files from local filesystem.  Defaults to
    /// static assets if not passed as CLI args.
    fn net_config(&self) -> NetConfig {
        fn read_config(party_idx: usize, path: &Option<String>) -> NodeConfig {
            let path_to_config = match path {
                Some(path) => path.clone(),
                None => get_path_to_local_node_config(NODE_CONFIG_KIND_MAIN, 0, &party_idx)
                    .to_string_lossy()
                    .to_string(),
            };

            read_node_config(Path::new(&path_to_config)).unwrap()
        }

        [
            read_config(0, self.path_to_node_0_config()),
            read_config(1, self.path_to_node_1_config()),
            read_config(2, self.path_to_node_2_config()),
        ]
    }
}

impl From<&CliOptions> for requests::Dispatcher {
    fn from(options: &CliOptions) -> Self {
        Self::new(requests::DispatcherOptions::from(options))
    }
}

impl From<&CliOptions> for requests::DispatcherOptions {
    fn from(_options: &CliOptions) -> Self {
        Self::new()
    }
}

impl From<&CliOptions> for requests::Factory {
    fn from(options: &CliOptions) -> Self {
        Self::new(requests::FactoryOptions::from(options))
    }
}

impl From<&CliOptions> for requests::FactoryOptions {
    fn from(_options: &CliOptions) -> Self {
        // TODD: hydrate batch-profile from cli options.
        Self::new(requests::BatchProfile::Simple(UNIQUENESS_MESSAGE_TYPE))
    }
}

impl From<&CliOptions> for requests::BatchGenerator {
    fn from(options: &CliOptions) -> Self {
        Self::new(
            requests::BatchGeneratorOptions::from(options),
            requests::Factory::from(options),
        )
    }
}

impl From<&CliOptions> for requests::BatchGeneratorOptions {
    fn from(options: &CliOptions) -> Self {
        Self::new(
            requests::BatchSize::Static(*options.batch_size()),
            *options.n_batches(),
        )
    }
}
