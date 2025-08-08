use super::constants::{FNAME_IRIS_CODES_1K, FNAME_IRIS_DELETIONS_1K};
use crate::utils::{
    constants::PARTY_IDX_SET,
    convertor, fsys,
    types::{NetConfig, PartyIdx},
};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodePair};
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::genesis::utils::aws::IrisDeletionsForS3;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIrisPairSet;
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::{IntoChunks, Itertools};
use serde_json;
use serde_json::Deserializer;
use std::{
    fs::{self, File},
    io::{BufReader, Error},
};

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub fn read_iris_codes(
    n_to_read: usize,
    skip_offset: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    let path_to_resources = fsys::get_assets_path(FNAME_IRIS_CODES_1K);

    Ok(
        Deserializer::from_reader(BufReader::new(File::open(path_to_resources).unwrap()))
            .into_iter::<Base64IrisCode>()
            .skip(skip_offset)
            .map(|x| IrisCode::from(&x.unwrap()))
            .tuples()
            .take(n_to_read),
    )
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris code pairs.
///
pub fn read_iris_codes_batch(
    batch_size: usize,
    n_to_read: usize,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    Ok(read_iris_codes(n_to_read, skip_offset)
        .unwrap()
        .chunks(batch_size))
}

/// Returns serial identifiers associated with deleted Iris's.
///
/// # Arguments
///
/// * `n_to_read` - Number of deletions to read into memory.
/// * `skip_offset` - Offset from which to start reading deletions.
///
/// # Returns
///
/// Vec of serial identifiers associated with deleted Iris's.
///
pub fn read_iris_deletions(
    n_to_read: usize,
    skip_offset: usize,
) -> Result<Vec<IrisSerialId>, Error> {
    let path_to_resource = fsys::get_assets_path(FNAME_IRIS_DELETIONS_1K);
    let IrisDeletionsForS3 { deleted_serial_ids } =
        serde_json::from_str(&std::fs::read_to_string(path_to_resource)?)?;

    Ok(deleted_serial_ids
        .into_iter()
        .skip(skip_offset)
        .take(n_to_read)
        .collect())
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris shares.
///
pub fn read_iris_shares(
    n_to_read: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    Ok(read_iris_codes(n_to_read, skip_offset)
        .unwrap()
        .map(move |code_pair| convertor::to_galois_ring_share_pair_set(rng_state, &code_pair)))
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `n_to_read` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris shares.
///
pub fn read_iris_shares_batch(
    batch_size: usize,
    n_to_read: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    Ok(read_iris_shares(n_to_read, rng_state, skip_offset)
        .unwrap()
        .chunks(batch_size))
}

/// Returns network configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `kind` - Kind of node configuration toml file to be read into memory.
/// * `idx` - Ordinal identifier of node configuration toml file to be read into memory.
///
/// # Returns
///
/// Network level configuration.
///
pub fn read_net_config(kind: &str, idx: usize) -> Result<NetConfig, Error> {
    Ok(PARTY_IDX_SET
        .iter()
        .map(|party_idx| read_node_config(party_idx, kind, idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap())
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
    let path_to_resource = fsys::get_assets_path(
        format!(
            "node-config/{}/{}.toml",
            fsys::get_exec_env_subdirectory(),
            fname
        )
        .as_str(),
    );

    Ok(toml::from_str(&fs::read_to_string(path_to_resource).unwrap()).unwrap())
}

#[cfg(test)]
mod tests {
    use super::super::constants::NODE_CONFIG_KIND_GENESIS;
    use super::{
        read_iris_codes, read_iris_codes_batch, read_iris_deletions, read_iris_shares,
        read_iris_shares_batch, read_net_config, read_node_config, read_node_config_by_name,
    };
    use crate::utils::constants::{PARTY_COUNT, PARTY_IDX_SET};

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_read_iris_code_pairs() {
        for (n_to_read, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in read_iris_codes(n_to_read, skip_offset).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_read_iris_code_pairs_batch() {
        for (batch_size, n_to_read, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in read_iris_codes_batch(batch_size, n_to_read, skip_offset)
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
    fn test_read_iris_deletions() {
        for (n_take, skip_offset) in [(2, 0), (10, 2)] {
            let n_read = read_iris_deletions(n_take, skip_offset).unwrap().len();
            assert_eq!(n_read, n_take);
        }
    }

    #[test]
    fn test_read_iris_shares() {
        for (n_to_read, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for shares in read_iris_shares(n_to_read, DEFAULT_RNG_STATE, skip_offset).unwrap() {
                n_read += 1;
                assert_eq!(shares.len(), PARTY_COUNT);
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_read_iris_shares_batch() {
        for (batch_size, n_to_read, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in
                read_iris_shares_batch(batch_size, n_to_read, DEFAULT_RNG_STATE, skip_offset)
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
    fn test_read_net_config() {
        let net_config = read_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap();
        assert!(net_config.len() == PARTY_COUNT);
        for (party_idx, node_config) in net_config.iter().enumerate() {
            assert!(node_config.party_id == party_idx);
        }
    }

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
