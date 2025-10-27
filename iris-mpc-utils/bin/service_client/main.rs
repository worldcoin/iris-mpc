use std::path::Path;

use clap::Parser;
use eyre::Result;

use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_utils::{
    state::fsys::{
        local::get_path_to_node_config as get_path_to_local_node_config, reader::read_node_config,
    },
    types::NetConfig,
};

mod request_dispatcher;
mod request_generator;

#[derive(Debug, Parser, Clone)]
struct Options {
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

impl Options {
    fn path_to_node_0_config(&self) -> &Option<String> {
        &self.path_to_node_0_config
    }

    fn path_to_node_1_config(&self) -> &Option<String> {
        &self.path_to_node_1_config
    }

    fn path_to_node_2_config(&self) -> &Option<String> {
        &self.path_to_node_2_config
    }

    fn net_config(&self) -> NetConfig {
        fn load_config(party_idx: usize, path: &Option<String>) -> NodeConfig {
            const DEFAULT_NODE_CONFIG_KIND: &str = "main";
            const DEFAULT_NODE_CONFIG_IDX: usize = 0;

            let path_to_config = match path {
                Some(path) => path.to_owned(),
                None => get_path_to_local_node_config(
                    DEFAULT_NODE_CONFIG_KIND,
                    DEFAULT_NODE_CONFIG_IDX,
                    &party_idx,
                )
                .to_string_lossy()
                .to_string(),
            };

            read_node_config(Path::new(&path_to_config)).unwrap()
        }

        [
            load_config(0, self.path_to_node_0_config()),
            load_config(1, self.path_to_node_1_config()),
            load_config(2, self.path_to_node_2_config()),
        ]
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    let options = Options::parse();
    println!("Running with options: {:?}", &options,);

    Ok(())
}
