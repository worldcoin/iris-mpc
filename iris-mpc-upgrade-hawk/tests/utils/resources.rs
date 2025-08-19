use super::{TestRunContextInfo, TestRunEnvironment};
use iris_mpc_common::config::Config;
use std::io::Error;
use std::path::{Path, PathBuf};

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
pub fn get_resources_root() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/resources")
}

/// Returns the path in the source tree of a resource asset.
pub fn get_resource_path(location: &str) -> PathBuf {
    Path::new(&get_resources_root()).join(location)
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
