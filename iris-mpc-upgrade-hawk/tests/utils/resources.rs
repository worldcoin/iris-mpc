use iris_mpc_common::config::Config as NodeConfig;
use std::io::Error;

/// Returns path to resources root directory.
fn get_path_to_resources() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

/// Returns node configuration deserialized from a toml file.
pub fn read_node_config(config_fname: String) -> Result<NodeConfig, Error> {
    let path_to_cfg = format!(
        "{}/node-config/{}.toml",
        get_path_to_resources(),
        config_fname
    );
    let cfg = std::fs::read_to_string(path_to_cfg)?;

    Ok(toml::from_str(&cfg).unwrap())
}
