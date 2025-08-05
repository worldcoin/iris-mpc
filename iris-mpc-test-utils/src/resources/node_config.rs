use crate::utils::{
    fsys::{get_path_to_assets, get_subdirectory_of_exec_env},
    types::PartyIdx,
};
use iris_mpc_common::config::Config as NodeConfig;
use std::io::Error;

/// Node config kind: genesis.
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";

/// Returns a node's config deserialized from environment variables.
///
/// # Returns
///
/// A node's config deserialized from environment variables.
///
pub fn generate_node_config_from_env_vars() -> NodeConfig {
    // Activate environment variables.
    dotenvy::dotenv().ok();

    NodeConfig::load_config("SMPC").unwrap()
}

/// Returns node configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `party_idx` - Ordinal identifier of MPC participant.
/// * `config_kind` - Kind of node configuration toml file to be read into memory.
/// * `config_idx` - Ordinal identifier of node configuration toml file to be read into memory.
///
/// # Returns
///
/// A node configuration file.
///
pub fn read_node_config(
    party_idx: &PartyIdx,
    config_kind: &str,
    config_idx: usize,
) -> Result<NodeConfig, Error> {
    let fname = format!("node-{}-{}-{}", party_idx, config_kind, config_idx);

    read_node_config_by_name(fname)
}

/// Returns node configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `fname` - Node configuration file name.
///
/// # Returns
///
/// A node configuration file.
///
pub fn read_node_config_by_name(fname: String) -> Result<NodeConfig, Error> {
    // Set path.
    let path_to_resource = format!(
        "{}/node-config/{}/{}.toml",
        get_path_to_assets(),
        get_subdirectory_of_exec_env(),
        fname
    );

    // Set raw config file content.
    let cfg = std::fs::read_to_string(path_to_resource)?;

    Ok(toml::from_str(&cfg).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{read_node_config, read_node_config_by_name, NODE_CONFIG_KIND_GENESIS};
    use crate::utils::constants::PARTY_IDX_SET;

    #[test]
    fn test_read_node_config() {
        let config_idx = 0;
        PARTY_IDX_SET.iter().for_each(|party_idx| {
            let cfg = read_node_config(&party_idx, NODE_CONFIG_KIND_GENESIS, config_idx).unwrap();
            assert!(cfg.party_id == *party_idx);
        });
    }

    #[test]
    fn test_read_node_config_by_name() {
        PARTY_IDX_SET.iter().for_each(|party_idx| {
            let fname = format!("node-{}-genesis-0", party_idx);
            let cfg = read_node_config_by_name(fname).unwrap();
            assert!(cfg.party_id == *party_idx);
        });
    }
}
