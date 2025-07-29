use super::{
    constants::COUNT_OF_PARTIES, convertor::to_galois_ring_shares, types::GaloisRingSharedIrisPair,
    IrisCodePair, TestRunContextInfo, TestRunEnvironment,
};
use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::{IntoChunks, Itertools};
use serde_json;
use std::{fs::File, io::BufReader, io::Error};

/// Returns subdirectory name for current test run environment.
fn get_subdirectory_of_env(env: &TestRunEnvironment) -> &'static str {
    match env {
        TestRunEnvironment::Docker => "docker",
        TestRunEnvironment::Local => "local",
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
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
/// * `max_items` - Maximum number of Iris code pairs to read.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub fn read_iris_code_pairs(
    skip_offset: usize,
    max_items: usize,
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
        .take(max_items);

    Ok(stream)
}

/// Returns chunked iterator over Iris code pairs deserialized from an ndjson file.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
/// * `max_items` - Maximum number of Iris code pairs to read.
///
/// # Returns
///
/// A chunked iterator over Iris code pairs.
///
pub fn read_iris_code_pairs_batch(
    batch_size: usize,
    skip_offset: usize,
    max_items: usize,
) -> Result<IntoChunks<impl Iterator<Item = IrisCodePair>>, Error> {
    let stream = read_iris_code_pairs(skip_offset, max_items)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
/// * `max_items` - Maximum number of Iris code pairs to read.
///
/// # Returns
///
/// An iterator over Iris shares.
///
pub fn read_iris_shares(
    rng_state: u64,
    skip_offset: usize,
    max_items: usize,
) -> Result<impl Iterator<Item = Box<[GaloisRingSharedIrisPair; COUNT_OF_PARTIES]>>, Error> {
    let stream = read_iris_code_pairs(skip_offset, max_items)
        .unwrap()
        .map(move |code_pair| to_galois_ring_shares(rng_state, &code_pair));

    Ok(stream)
}

/// Returns chunked iterator over Iris shares deserialized from a stream of Iris Code pairs.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
/// * `max_items` - Maximum number of Iris code pairs to read.
///
/// # Returns
///
/// A chunked iterator over Iris shares.
///
pub fn read_iris_shares_batch(
    batch_size: usize,
    rng_state: u64,
    skip_offset: usize,
    max_items: usize,
) -> Result<
    IntoChunks<impl Iterator<Item = Box<[GaloisRingSharedIrisPair; COUNT_OF_PARTIES]>>>,
    Error,
> {
    let stream = read_iris_shares(rng_state, skip_offset, max_items)
        .unwrap()
        .chunks(batch_size);

    Ok(stream)
}

/// Returns node configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `ctx` - Contextual information associated with a test run.
/// * `config_fname` - File name of node configuration toml file being read into memory.
///
/// # Returns
///
/// A node configuration file.
///
pub fn read_node_config(
    ctx: &TestRunContextInfo,
    config_fname: String,
) -> Result<NodeConfig, Error> {
    // Set path.
    let path_to_resource = format!(
        "{}/node-config/{}/{}.toml",
        get_path_to_resources(),
        get_subdirectory_of_env(ctx.env()),
        config_fname
    );

    // Set raw config file content.
    let cfg = std::fs::read_to_string(path_to_resource)?;

    Ok(toml::from_str(&cfg).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{
        get_path_to_resources, get_subdirectory_of_env, read_iris_code_pairs,
        read_iris_code_pairs_batch, read_iris_shares, read_iris_shares_batch, read_node_config,
        TestRunContextInfo, TestRunEnvironment, COUNT_OF_PARTIES,
    };
    use std::path::Path;

    const DEFAULT_RNG_STATE: u64 = 93;

    impl TestRunContextInfo {
        fn new_1() -> Self {
            Self::new(100, 1)
        }
    }

    #[test]
    fn test_get_subdirectory_of_env() {
        for (subdir, env) in [
            ("docker", TestRunEnvironment::Docker),
            ("local", TestRunEnvironment::Local),
        ] {
            assert_eq!(subdir, get_subdirectory_of_env(&env));
        }
    }

    #[test]
    fn test_get_path_to_resources() {
        assert!(Path::new(&get_path_to_resources()).exists());
    }

    #[test]
    fn test_read_iris_code_pairs() {
        // NOTE: currently runs against a default ndjson file of 1000 iris codes (i.e. 500 pairs).
        for (skip_offset, max_items) in [(0, 100), (838, 81)] {
            let mut n_read = 0;
            for _ in read_iris_code_pairs(skip_offset, max_items).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_read, max_items);
        }
    }

    #[test]
    fn test_read_iris_code_pairs_batch() {
        // NOTE: currently runs against a default ndjson file of 1000 iris codes (i.e. 500 pairs).
        for (skip_offset, max_items, batch_size, expected_batches) in
            [(0, 100, 10, 10), (838, 81, 9, 9)]
        {
            let mut n_chunks = 0;
            for chunk in read_iris_code_pairs_batch(batch_size, skip_offset, max_items)
                .unwrap()
                .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for _ in chunk.into_iter() {
                    n_items += 1;
                }
                assert_eq!(n_items, batch_size);
            }
            assert_eq!(n_chunks, expected_batches);
        }
    }

    #[test]
    fn test_read_iris_shares() {
        for (skip_offset, max_items) in [(0, 100), (838, 81)] {
            let mut n_read = 0;
            for shares in read_iris_shares(DEFAULT_RNG_STATE, skip_offset, max_items).unwrap() {
                n_read += 1;
                assert_eq!(shares.len(), COUNT_OF_PARTIES);
            }
            assert_eq!(n_read, max_items);
        }
    }

    #[test]
    fn test_read_iris_shares_batch() {
        for (skip_offset, max_items, batch_size, expected_batches) in
            [(0, 100, 10, 10), (838, 81, 9, 9)]
        {
            let mut n_chunks = 0;
            for chunk in
                read_iris_shares_batch(batch_size, DEFAULT_RNG_STATE, skip_offset, max_items)
                    .unwrap()
                    .into_iter()
            {
                n_chunks += 1;
                let mut n_items = 0;
                for item in chunk.into_iter() {
                    assert_eq!(item.len(), COUNT_OF_PARTIES);
                    n_items += 1;
                }
                assert_eq!(n_items, batch_size);
            }
            assert_eq!(n_chunks, expected_batches);
        }
    }

    #[test]
    fn test_read_node_config() {
        let ctx = TestRunContextInfo::new_1();
        for party_id in [0, 1, 2] {
            let cfg_fname = format!("node-{}-genesis-0", party_id);
            let cfg = read_node_config(&ctx, cfg_fname).unwrap();
            assert!(cfg.party_id == party_id);
        }
    }
}
