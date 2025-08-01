use super::{convertor::to_galois_ring_shares, TestRunExecutionEnvironment};
use iris_mpc_common::{
    config::Config as NodeConfig,
    iris_db::iris::{IrisCode, IrisCodePair},
    PartyIdx,
};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIrisPairSet;
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::{IntoChunks, Itertools};
use serde_json;
use std::{fs::File, io::BufReader, io::Error};

/// Node config kind: genesis.
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";

/// Returns subdirectory name for current test run environment.
fn get_subdirectory_of_exec_env() -> &'static str {
    match TestRunExecutionEnvironment::new() {
        TestRunExecutionEnvironment::Docker => "docker",
        TestRunExecutionEnvironment::Local => "local",
    }
}

/// Returns path to resources root directory.
fn get_path_to_resources() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

/// Returns iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub fn read_iris_code_pairs(
    read_maximum: usize,
    skip_offset: usize,
) -> Result<impl Iterator<Item = IrisCodePair>, Error> {
    // Set path.
    // TODO: use strong names for ndjson file.
    let path_to_iris_codes = format!(
        "{}/iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson",
        get_path_to_resources(),
    );

    // Set file stream.
    let file = File::open(path_to_iris_codes).unwrap();
    let reader = BufReader::new(file);
    let stream = serde_json::Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(skip_offset)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(read_maximum);

    Ok(stream)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris code pairs.
///
pub fn read_iris_code_pairs_batch(
    batch_size: usize,
    read_maximum: usize,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let stream = read_iris_code_pairs(read_maximum, skip_offset)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// An iterator over Iris shares.
///
pub fn read_iris_shares(
    read_maximum: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>, Error> {
    let stream = read_iris_code_pairs(read_maximum, skip_offset)
        .unwrap()
        .map(move |code_pair| Box::new(to_galois_ring_shares(rng_state, &code_pair)));

    Ok(stream)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `read_maximum` - Maximum number of Iris code pairs to read.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
///
/// # Returns
///
/// A chunked iterator over Iris shares.
///
pub fn read_iris_shares_batch(
    batch_size: usize,
    read_maximum: usize,
    rng_state: u64,
    skip_offset: usize,
) -> Result<IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>>, Error> {
    let stream = read_iris_shares(read_maximum, rng_state, skip_offset)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
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
        get_path_to_resources(),
        get_subdirectory_of_exec_env(),
        fname
    );

    // Set raw config file content.
    let cfg = std::fs::read_to_string(path_to_resource)?;

    Ok(toml::from_str(&cfg).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{
        get_path_to_resources, get_subdirectory_of_exec_env, read_iris_code_pairs,
        read_iris_code_pairs_batch, read_iris_shares, read_iris_shares_batch, read_node_config,
        read_node_config_by_name, TestRunExecutionEnvironment,
    };
    use iris_mpc_common::{PARTY_COUNT, PARTY_IDX_SET};
    use std::path::Path;

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_get_subdirectory_of_exec_env() {
        match TestRunExecutionEnvironment::new() {
            TestRunExecutionEnvironment::Docker => {
                assert_eq!(get_subdirectory_of_exec_env(), "docker");
            }
            TestRunExecutionEnvironment::Local => {
                assert_eq!(get_subdirectory_of_exec_env(), "local");
            }
        };
    }

    #[test]
    fn test_get_path_to_resources() {
        assert!(Path::new(&get_path_to_resources()).exists());
    }

    #[test]
    fn test_read_iris_code_pairs() {
        // NOTE: currently runs against a default ndjson file of 1000 iris codes (i.e. 500 pairs).
        for (read_maximum, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for _ in read_iris_code_pairs(read_maximum, skip_offset).unwrap() {
                n_read += 1;
            }
            assert_eq!(read_maximum, n_read);
        }
    }

    #[test]
    fn test_read_iris_code_pairs_batch() {
        // NOTE: currently runs against a default ndjson file of 1000 iris codes (i.e. 500 pairs).
        for (batch_size, read_maximum, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in read_iris_code_pairs_batch(batch_size, read_maximum, skip_offset)
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
    fn test_read_iris_shares() {
        for (read_maximum, skip_offset) in [(100, 0), (81, 838)] {
            let mut n_read = 0;
            for shares in read_iris_shares(read_maximum, DEFAULT_RNG_STATE, skip_offset).unwrap() {
                n_read += 1;
                assert_eq!(shares.len(), PARTY_COUNT);
            }
            assert_eq!(read_maximum, n_read);
        }
    }

    #[test]
    fn test_read_iris_shares_batch() {
        for (batch_size, read_maximum, skip_offset, expected_batches) in
            [(10, 100, 0, 10), (9, 81, 838, 9)]
        {
            let mut n_chunks = 0;
            for chunk in
                read_iris_shares_batch(batch_size, read_maximum, DEFAULT_RNG_STATE, skip_offset)
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
