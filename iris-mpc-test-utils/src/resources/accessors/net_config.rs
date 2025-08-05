use super::node_config::read_node_config;
use crate::utils::{constants::PARTY_IDX_SET, types::NetConfig};
use std::io::Error;

/// Returns network configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `config_kind` - Kind of node configuration toml file to be read into memory.
/// * `config_idx` - Ordinal identifier of node configuration toml file to be read into memory.
///
/// # Returns
///
/// Network level configuration.
///
pub fn read_net_config(config_kind: &str, config_idx: usize) -> Result<NetConfig, Error> {
    Ok(PARTY_IDX_SET
        .iter()
        .map(|party_idx| read_node_config(party_idx, config_kind, config_idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap())
}

#[cfg(test)]
mod tests {
    use super::{super::NODE_CONFIG_KIND_GENESIS, read_net_config};
    use crate::utils::constants::PARTY_COUNT;

    #[test]
    fn test_read_net_config() {
        let net_config = read_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap();
        assert!(net_config.len() == PARTY_COUNT);
        for (party_idx, node_config) in net_config.iter().enumerate() {
            assert!(node_config.party_id == party_idx);
        }
    }
}
