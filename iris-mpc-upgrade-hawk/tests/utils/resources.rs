use super::{
    constants::COUNT_OF_PARTIES, types::GaloisRingSharedIrisPair, IrisCodePair,
    TestExecutionEnvironment, TestRunContextInfo,
};
use iris_mpc_common::{
    config::Config as NodeConfig, iris_db::iris::IrisCode, IrisSerialId, IrisVersionId,
};
use iris_mpc_cpu::{
    protocol::shared_iris::GaloisRingSharedIris, py_bindings::plaintext_store::Base64IrisCode,
};
use itertools::{IntoChunks, Itertools};
use rand::{rngs::StdRng, SeedableRng};
use serde_json;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Error},
};

/// Returns subdirectory name for current test run environment.
fn get_subdirectory_of_exec_env(exec_env: &TestExecutionEnvironment) -> &'static str {
    match exec_env {
        TestExecutionEnvironment::Docker => "docker",
        TestExecutionEnvironment::Local => "local",
    }
}

/// Returns path to resources root directory.
fn get_path_to_resources() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

pub fn read_plaintext_iris(
    skip_offset: usize,
    max_items: usize,
) -> Result<Vec<IrisCodePair>, Error> {
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

    Ok(stream.collect())
}

pub fn get_genesis_input(
    pairs: &[IrisCodePair],
) -> HashMap<IrisSerialId, (IrisVersionId, IrisCode, IrisCode)> {
    let mut r = HashMap::new();
    for (idx, (left, right)) in pairs.iter().enumerate() {
        r.insert(idx as _, (0, left.clone(), right.clone()));
    }
    r
}

pub fn encode_plaintext_iris_for_party(
    pairs: &[IrisCodePair],
    rng_state: u64,
    party_idx: usize,
) -> Vec<GaloisRingSharedIrisPair> {
    pairs
        .iter()
        .map(|code_pair| {
            // Set RNG for each pair to match shares_encoding.rs behavior
            let mut shares_seed = StdRng::seed_from_u64(rng_state);

            // Set MPC participant specific Iris shares from Iris code + entropy.
            let (code_l, code_r) = code_pair;
            let shares_l =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_l.to_owned());
            let shares_r =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_r.to_owned());

            (shares_l[party_idx].clone(), shares_r[party_idx].clone())
        })
        .collect()
}

/// Returns node configuration deserialized from a toml file.
///
/// # Arguments
///
/// * `exec_env` - Contextual information associated with a test run.
/// * `config_fname` - File name of node configuration toml file being read into memory.
///
/// # Returns
///
/// A node configuration file.
///
pub fn read_node_config(
    exec_env: &TestExecutionEnvironment,
    config_fname: String,
) -> Result<NodeConfig, Error> {
    // Set path.
    let path_to_resource = format!(
        "{}/node-config/{}/{}.toml",
        get_path_to_resources(),
        get_subdirectory_of_exec_env(exec_env),
        config_fname
    );

    // Set raw config file content.
    let cfg = std::fs::read_to_string(path_to_resource)?;

    Ok(toml::from_str(&cfg).unwrap())
}

#[cfg(test)]
mod tests {
    use super::{
        get_path_to_resources, get_subdirectory_of_exec_env, read_node_config, read_plaintext_iris,
        TestExecutionEnvironment, TestRunContextInfo, COUNT_OF_PARTIES,
    };
    use std::path::Path;

    const DEFAULT_RNG_STATE: u64 = 93;

    impl TestRunContextInfo {
        fn new_1() -> Self {
            Self::new(100, 1)
        }
    }

    #[test]
    fn test_get_subdirectory_of_exec_env() {
        for (subdir, env) in [
            ("docker", TestExecutionEnvironment::Docker),
            ("local", TestExecutionEnvironment::Local),
        ] {
            assert_eq!(subdir, get_subdirectory_of_exec_env(&env));
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
            for _ in read_plaintext_iris(skip_offset, max_items).unwrap() {
                n_read += 1;
            }
            assert_eq!(n_read, max_items);
        }
    }

    #[test]
    fn test_read_node_config() {
        let ctx = TestRunContextInfo::new_1();
        for party_id in [0, 1, 2] {
            let cfg_fname = format!("node-{}-genesis-0", party_id);
            let cfg = read_node_config(ctx.exec_env(), cfg_fname).unwrap();
            assert!(cfg.party_id == party_id);
        }
    }
}
