use iris_mpc_common::config::Config as NodeConfig;
use std::io::Error;

/// Returns node configuration deserialized from a toml file.
pub fn read_node_config(idx_of_node: usize) -> Result<NodeConfig, Error> {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    let path_to_cfg = format!("{crate_root}/tests/resources/node-{}.toml", idx_of_node);
    let cfg = std::fs::read_to_string(path_to_cfg)?;

    Ok(toml::from_str(&cfg).unwrap())
}
