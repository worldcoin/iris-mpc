use crate::utils::{
    constants::PARTY_IDX_SET,
    convertor, fsys,
    types::{GaloisRingSharedIrisPairSet, IrisCodePair, NetConfig, PartyIdx},
};
use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::{IntoChunks, Itertools};
use serde_json::{self, Deserializer};
use std::{
    fs::{self, File},
    io::{BufReader, Error},
    path::Path,
};

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
pub fn load_iris_codes(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    let reader = BufReader::new(File::open(path_to_codes).unwrap());
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(n_to_skip)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(n_to_read);

    Ok(stream)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
pub fn load_iris_codes_batch(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let stream = load_iris_codes(path_to_codes, n_to_read, n_to_skip)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn load_iris_shares(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    rng_state: u64,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    let stream = load_iris_codes(path_to_codes, n_to_read, n_to_skip)
        .unwrap()
        .map(move |code_pair| convertor::to_galois_ring_share_pair_set(rng_state, &code_pair));

    Ok(stream)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
pub fn load_iris_shares_batch(
    path_to_codes: &Path,
    n_to_read: usize,
    n_to_skip: usize,
    batch_size: usize,
    rng_state: u64,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    let stream = load_iris_shares(path_to_codes, n_to_read, n_to_skip, rng_state)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns network configuration deserialized from a toml file.
pub fn load_net_config(kind: &str, idx: usize) -> Result<NetConfig, Error> {
    let config = PARTY_IDX_SET
        .iter()
        .map(|party_idx| load_node_config(party_idx, kind, idx).unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    Ok(config)
}

/// Returns node configuration deserialized from a toml file.
pub fn load_node_config(party_idx: &PartyIdx, kind: &str, idx: usize) -> Result<NodeConfig, Error> {
    let fname = format!("node-{}-{}-{}", party_idx, kind, idx);

    load_node_config_by_name(fname)
}

/// Returns node configuration deserialized from a toml file.
pub fn load_node_config_by_name(fname: String) -> Result<NodeConfig, Error> {
    Ok(toml::from_str(
        &fs::read_to_string(fsys::get_path_to_asset(
            format!(
                "node-config/{}/{}.toml",
                fsys::get_execution_host_subdirectory(),
                fname
            )
            .as_str(),
        ))
        .unwrap(),
    )
    .unwrap())
}

#[cfg(test)]
mod tests {
    use super::super::constants::FNAME_IRIS_CODES_1K;
    use super::{
        load_iris_codes, load_iris_codes_batch, load_iris_shares, load_iris_shares_batch,
        load_net_config, load_node_config, load_node_config_by_name,
    };
    use crate::utils::{
        constants::{NODE_CONFIG_KIND_GENESIS, PARTY_COUNT, PARTY_IDX_SET},
        fsys,
    };

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_iris_codes_exist() {
        let path_to_codes = fsys::get_path_to_asset(FNAME_IRIS_CODES_1K);

        assert!(path_to_codes.exists())
    }

    #[test]
    fn test_load_iris_codes() {
        let path_to_codes = fsys::get_path_to_asset(FNAME_IRIS_CODES_1K);

        for (n_to_read, n_to_skip) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in load_iris_codes(&path_to_codes, n_to_read, n_to_skip).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_load_iris_codes_batch() {
        let path_to_codes = fsys::get_path_to_asset(FNAME_IRIS_CODES_1K);

        for (batch_size, n_to_read, n_to_skip, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in load_iris_codes_batch(&path_to_codes, n_to_read, n_to_skip, batch_size)
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
        let path_to_codes = fsys::get_path_to_asset(FNAME_IRIS_CODES_1K);

        for (n_to_read, n_to_skip) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for shares in
                load_iris_shares(&path_to_codes, n_to_read, n_to_skip, DEFAULT_RNG_STATE).unwrap()
            {
                n_read += 1;
                assert_eq!(shares.len(), PARTY_COUNT);
            }
            assert_eq!(n_to_read, n_read);
        }
    }

    #[test]
    fn test_load_iris_shares_batch() {
        let path_to_codes = fsys::get_path_to_asset(FNAME_IRIS_CODES_1K);

        for (batch_size, n_to_read, n_to_skip, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in load_iris_shares_batch(
                &path_to_codes,
                n_to_read,
                n_to_skip,
                batch_size,
                DEFAULT_RNG_STATE,
            )
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
    fn test_load_net_config() {
        let net_config = load_net_config(NODE_CONFIG_KIND_GENESIS, 0).unwrap();
        assert!(net_config.len() == PARTY_COUNT);
        for (party_idx, node_config) in net_config.iter().enumerate() {
            assert!(node_config.party_id == party_idx);
        }
    }

    #[test]
    fn test_load_node_config() {
        let config_idx = 0;
        PARTY_IDX_SET.iter().for_each(|party_idx| {
            let cfg = load_node_config(party_idx, NODE_CONFIG_KIND_GENESIS, config_idx).unwrap();
            assert!(cfg.party_id == *party_idx);
        });
    }

    #[test]
    fn test_load_node_config_by_name() {
        PARTY_IDX_SET.iter().for_each(|party_idx| {
            let fname = format!("node-{}-genesis-0", party_idx);
            let cfg = load_node_config_by_name(fname).unwrap();
            assert!(cfg.party_id == *party_idx);
        });
    }
}
