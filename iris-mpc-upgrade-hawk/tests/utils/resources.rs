use crate::utils::IrisCodePair;

use super::{TestRunContextInfo, TestRunEnvironment};
use iris_mpc_common::config::Config;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::py_bindings::plaintext_store::Base64IrisCode;
use itertools::Itertools;
use std::fs::File;
use std::io::{BufReader, Error};

impl TestRunEnvironment {
    /// Returns subdirectory name for current test run environment.
    pub fn subdirectory(&self) -> &str {
        match self {
            TestRunEnvironment::Docker => "docker",
            TestRunEnvironment::Local => "local",
        }
    }
}

/// Returns path to resources root directory.
fn get_resources_root() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

/// Returns the path in the source tree of a resource asset.
///
/// Format of resource location should start with a leading forward slash.
pub fn get_resource_path(location: String) -> String {
    format!("{}{}", get_resources_root(), location)
}

/// Returns node configuration deserialized from a toml file.
pub fn read_node_config(ctx: &TestRunContextInfo, config_fname: String) -> Result<Config, Error> {
    let path_to_cfg = format!(
        "{}/node-config/{}/{}.toml",
        get_resources_root(),
        ctx.env().subdirectory(),
        config_fname
    );
    let cfg = std::fs::read_to_string(path_to_cfg)?;

    Ok(toml::from_str(&cfg).unwrap())
}

pub fn read_plaintext_iris(
    skip_offset: usize,
    max_items: usize,
) -> Result<Vec<IrisCodePair>, Error> {
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
