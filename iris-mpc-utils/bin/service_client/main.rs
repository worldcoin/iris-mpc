use std::path::Path;

use clap::Parser;
use eyre::Result;

use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_utils::{
    constants::NODE_CONFIG_KIND_MAIN,
    state::fsys::{
        local::get_path_to_node_config as get_path_to_local_node_config, reader::read_node_config,
    },
    types::NetConfig,
};

mod requests;
mod responses;

#[derive(Debug, Parser, Clone)]
struct CliOptions {
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
    const DEFAULT_NODE_CONFIG_KIND: &str = NODE_CONFIG_KIND_MAIN;
    const DEFAULT_NODE_CONFIG_IDX: usize = 0;

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
                None => get_path_to_local_node_config(
                    CliOptions::DEFAULT_NODE_CONFIG_KIND,
                    CliOptions::DEFAULT_NODE_CONFIG_IDX,
                    &party_idx,
                )
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

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = CliOptions::parse();
    println!("Running with options: {:?}", &options,);

    Ok(())
}
