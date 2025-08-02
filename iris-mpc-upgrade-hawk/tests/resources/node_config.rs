use super::utils::{get_path_to_assets, get_subdirectory_of_exec_env};
use iris_mpc_common::{config::Config as NodeConfig, PartyIdx};
use std::io::Error;

/// Node config kind: genesis.
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";

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
    use super::{read_node_config, read_node_config_by_name};
    use iris_mpc_common::PARTY_IDX_SET;

    #[test]
    fn test_read_node_config() {
        for party_idx in PARTY_IDX_SET {
            for config_idx in [0] {
                let cfg = read_node_config(&party_idx, "genesis", config_idx).unwrap();
                assert!(cfg.party_id == party_idx);
            }
        }
    }

    #[test]
    fn test_read_node_config_by_name() {
        for party_idx in PARTY_IDX_SET {
            let fname = format!("node-{}-genesis-0", party_idx);
            let cfg = read_node_config_by_name(fname).unwrap();
            assert!(cfg.party_id == party_idx);
        }
    }
}
