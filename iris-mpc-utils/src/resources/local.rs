use super::reader;
use crate::{
    constants::PARTY_IDX,
    types::{GaloisRingSharedIrisPairSet, IrisCodePair, NetConfig, NodeExecutionHost, PartyIdx},
    utils::fsys,
};
use iris_mpc_common::config::Config as NodeConfig;
use itertools::IntoChunks;
use std::{io::Error, path::PathBuf};

/// Name of default local ndjson asset file.
const FNAME_NDJSON_1K: &str = "iris-codes-plaintext/20250710-1k.ndjson";

/// Returns iterator over default Iris code pairs deserialized from an ndjson file.
pub fn load_iris_codes(
    n_to_read: usize,
    n_to_skip: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    let path_to_codes = get_path_to_ndjson();

    reader::read_iris_codes(&path_to_codes, n_to_read, n_to_skip)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
pub fn load_iris_codes_batch(
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let path_to_codes = get_path_to_ndjson();

    reader::read_iris_codes_batch(&path_to_codes, n_to_read, n_to_skip, batch_size)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn load_iris_shares(
    n_to_read: usize,
    n_to_skip: usize,
    rng_state: u64,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    let path_to_codes = get_path_to_ndjson();

    reader::read_iris_shares(&path_to_codes, n_to_read, n_to_skip, rng_state)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn load_iris_shares_batch(
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
    rng_state: u64,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    let path_to_codes = get_path_to_ndjson();

    reader::read_iris_shares_batch(&path_to_codes, n_to_read, n_to_skip, batch_size, rng_state)
}

/// Returns a loaded node config file.
pub fn load_node_config(
    config_kind: &str,
    config_idx: usize,
    party_idx: &PartyIdx,
) -> Result<NodeConfig, Error> {
    let path_to_config = get_path_to_node_config(config_kind, config_idx, party_idx);

    reader::read_node_config(&path_to_config)
}

/// Returns network configuration deserialized from a toml file.
pub fn load_net_config(config_kind: &str, config_idx: usize) -> Result<NetConfig, Error> {
    let config = PARTY_IDX
        .iter()
        .map(|party_idx| load_node_config(config_kind, config_idx, party_idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    Ok(config)
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

/// Returns path to an asset within the crate assets sub-directory.
fn get_path_to_assets() -> PathBuf {
    fsys::get_path_to_subdir("assets")
}

/// Returns path to default Iris codes ndjson file.
fn get_path_to_ndjson() -> PathBuf {
    get_path_to_assets().join(FNAME_NDJSON_1K)
}

/// Returns path to a node config file.
fn get_path_to_node_config(config_kind: &str, config_idx: usize, party_idx: &PartyIdx) -> PathBuf {
    get_path_to_assets().join(
        format!(
            "node-config/{}/{config_kind}-{config_idx}-node-{party_idx}.toml",
            NodeExecutionHost::assets_subdirectory(),
        )
        .as_str(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        get_path_to_assets, get_path_to_ndjson, get_path_to_node_config, load_iris_codes,
        load_iris_codes_batch, load_iris_shares, load_iris_shares_batch, load_net_config,
        load_node_config,
    };
    use crate::constants::{NODE_CONFIG_KIND, NODE_CONFIG_KIND_GENESIS, PARTY_COUNT, PARTY_IDX};

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_get_path_to_assets() {
        assert!(get_path_to_assets().exists());
    }

    #[test]
    fn test_path_to_ndjson() {
        assert!(get_path_to_ndjson().exists());
    }

    #[test]
    fn test_path_to_node_config() {
        PARTY_IDX.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                assert!(get_path_to_node_config(kind, 0, party_idx).exists());
            });
        });
    }

    #[test]
    fn test_load_iris_codes() {
        for (n_to_read, n_to_skip) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in load_iris_codes(n_to_read, n_to_skip).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_load_iris_codes_batch() {
        for (batch_size, n_to_read, n_to_skip, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in load_iris_codes_batch(n_to_read, n_to_skip, batch_size)
                .unwrap()
                .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for _ in chunk.into_iter() {
                    n_items += 1;
                }
                assert_eq!(batch_size, n_items);
            }
            assert_eq!(expected_batches, n_chunks);
        }
    }

    #[test]
    fn test_load_iris_shares() {
        for (n_to_read, n_to_skip) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for shares in load_iris_shares(n_to_read, n_to_skip, DEFAULT_RNG_STATE).unwrap() {
                n_read += 1;
                assert_eq!(shares.len(), PARTY_COUNT);
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_load_iris_shares_batch() {
        for (batch_size, n_to_read, n_to_skip, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in load_iris_shares_batch(n_to_read, n_to_skip, batch_size, DEFAULT_RNG_STATE)
                .unwrap()
                .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for item in chunk.into_iter() {
                    assert_eq!(item.len(), PARTY_COUNT);
                    n_items += 1;
                }
                assert_eq!(batch_size, n_items);
            }
            assert_eq!(expected_batches, n_chunks);
        }
    }

    #[test]
    fn test_load_node_config() {
        PARTY_IDX.iter().for_each(move |party_idx| {
            NODE_CONFIG_KIND.iter().for_each(|kind| {
                load_node_config(kind, 0, party_idx).unwrap();
            });
        });
    }

    #[test]
    fn test_load_net_config() {
        let net_config = load_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap();
        assert!(net_config.len() == PARTY_COUNT);
        for (party_idx, node_config) in net_config.iter().enumerate() {
            assert!(node_config.party_id == party_idx);
        }
    }
}
