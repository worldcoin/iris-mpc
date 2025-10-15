use iris_mpc_common::config::Config as NodeConfig;
use std::{fs, io::Error, path::Path};
use toml;

/// Returns node configuration deserialized from a toml file.
pub fn read_node_config(path_to_config: &Path) -> Result<NodeConfig, Error> {
    assert!(path_to_config.exists());

    Ok(toml::from_str(&fs::read_to_string(path_to_config)?).unwrap())
}
