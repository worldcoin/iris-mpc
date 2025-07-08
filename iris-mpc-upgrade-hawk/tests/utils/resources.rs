use iris_mpc_common::config::Config as NodeConfig;
use std::io::Error;

/// Returns path to resources root directory.
fn get_path_to_resources() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

/// Returns node configuration deserialized from a toml file.
pub fn read_node_config(idx_of_node: usize) -> Result<NodeConfig, Error> {
    let path_to_cfg = format!(
        "{}/node-config/node-{}-genesis.toml",
        get_path_to_resources(),
        idx_of_node
    );
    let cfg = std::fs::read_to_string(path_to_cfg)?;

    Ok(toml::from_str(&cfg).unwrap())
}
