use super::reader;
use crate::{
    constants::PARTY_INDICES,
    types::{NodeConfigSet, NodeExecutionHost, PartyIdx},
};
use iris_mpc_common::config::Config as NodeConfig;
use std::{
    io::Error,
    path::{Path, PathBuf},
};

/// Returns path to an asset within the crate assets sub-directory.
fn get_path_to_assets() -> PathBuf {
    get_path_to_subdir("assets")
}

/// Returns path to a node config file.
pub fn get_path_to_node_config(
    config_kind: &str,
    config_idx: usize,
    party_idx: &PartyIdx,
) -> PathBuf {
    get_path_to_assets().join(
        format!(
            "node-config/{}/{config_kind}-{config_idx}-node-{party_idx}.toml",
            NodeExecutionHost::assets_subdirectory(),
        )
        .as_str(),
    )
}

/// Returns path to root directory.
pub fn get_path_to_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).into()
}

/// Returns path to sub-directory.
pub fn get_path_to_subdir(name: &str) -> PathBuf {
    get_path_to_root().join(name)
}

/// Returns a loaded node config file.
pub fn read_node_config(
    config_kind: &str,
    config_idx: usize,
    party_idx: &PartyIdx,
) -> Result<NodeConfig, Error> {
    let path_to_config = get_path_to_node_config(config_kind, config_idx, party_idx);

    reader::read_node_config(&path_to_config)
}

/// Returns network wide configuration deserialized from a set of toml files.
pub fn read_node_config_set(config_kind: &str, config_idx: usize) -> Result<NodeConfigSet, Error> {
    Ok(PARTY_INDICES
        .iter()
        .map(|party_idx| read_node_config(config_kind, config_idx, party_idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap())
}

impl NodeExecutionHost {
    /// Returns name of execution host specific assets subdirectory.
    pub(super) fn assets_subdirectory() -> &'static str {
        match NodeExecutionHost::default() {
            NodeExecutionHost::BareMetal => "baremetal",
            NodeExecutionHost::Docker => "docker",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        get_path_to_assets, get_path_to_node_config, get_path_to_root, get_path_to_subdir,
        read_node_config, read_node_config_set,
    };
    use crate::constants::{NODE_CONFIG_KIND, NODE_CONFIG_KIND_GENESIS, N_PARTIES, PARTY_INDICES};

    #[test]
    fn test_get_path_to_root() {
        assert!(get_path_to_root().exists());
    }

    #[test]
    fn test_get_path_to_subdir() {
        assert!(get_path_to_subdir("assets").exists());
    }

    #[test]
    fn test_get_path_to_assets() {
        assert!(get_path_to_assets().exists());
    }

    #[test]
    fn test_path_to_node_config() {
        PARTY_INDICES.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                assert!(get_path_to_node_config(kind, 0, party_idx).exists());
            });
        });
    }

    #[test]
    fn test_read_node_config() {
        PARTY_INDICES.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                read_node_config(kind, 0, party_idx).unwrap();
            });
        });
    }

    #[test]
    fn test_read_net_config() {
        let net_config = read_node_config_set(NODE_CONFIG_KIND_GENESIS, 0).unwrap();
        assert!(net_config.len() == N_PARTIES);
        for (party_idx, node_config) in net_config.iter().enumerate() {
            assert!(node_config.party_id == party_idx);
        }
    }
}
