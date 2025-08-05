use crate::utils::{
    fsys::{get_assets_root, get_exec_env_subdirectory},
    types::PartyIdx,
};
use iris_mpc_common::config::Config as NodeConfig;
use std::{fs, io::Error};

/// Node config kind: genesis.
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";

/// Returns a node's config deserialized from SMPC environment variables.
///
/// # Returns
///
/// A node's config deserialized from environment variables.
///
pub fn generate_node_config_from_env_vars() -> NodeConfig {
    // Activates environment variables.
    dotenvy::dotenv().ok();

    NodeConfig::load_config("SMPC").unwrap()
}

/// Returns node configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `party_idx` - Ordinal identifier of MPC participant.
/// * `kind` - Kind of node configuration toml file to be read into memory.
/// * `idx` - Ordinal identifier of node configuration toml file to be read into memory.
///
/// # Returns
///
/// A node configuration file.
///
pub fn read_node_config(party_idx: &PartyIdx, kind: &str, idx: usize) -> Result<NodeConfig, Error> {
    read_node_config_by_name(format!("node-{}-{}-{}", party_idx, kind, idx))
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
    let path_to_resource = format!(
        "{}/node-config/{}/{}.toml",
        get_assets_root(),
        get_exec_env_subdirectory(),
        fname
    );

    Ok(toml::from_str(&fs::read_to_string(path_to_resource).unwrap()).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{read_node_config, read_node_config_by_name, NODE_CONFIG_KIND_GENESIS};
    use crate::utils::constants::PARTY_IDX_SET;

    #[test]
    fn test_read_node_config() {
        let config_idx = 0;
        PARTY_IDX_SET.iter().for_each(|party_idx| {
            let cfg = read_node_config(party_idx, NODE_CONFIG_KIND_GENESIS, config_idx).unwrap();
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
