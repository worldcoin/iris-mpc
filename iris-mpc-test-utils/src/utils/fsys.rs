use super::types::ExecutionEnvironment;
use std::path::{Path, PathBuf};

/// Returns subdirectory name for current test run environment.
pub fn get_exec_env_subdirectory() -> &'static str {
    match ExecutionEnvironment::default() {
        ExecutionEnvironment::Docker => "docker",
        ExecutionEnvironment::Local => "local",
    }
}

/// Returns path to a resource within assets directory.
pub fn get_assets_path(subpath: String) -> PathBuf {
    Path::new(&get_assets_root()).join(subpath)
}

/// Returns path to assets directory.
pub fn get_assets_root() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/assets")
}

/// Returns path to a resource within data directory.
#[allow(dead_code)]
pub fn get_data_path(subpath: String) -> PathBuf {
    Path::new(&get_data_root()).join(subpath)
}

/// Returns path to data directory.
pub fn get_data_root() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/data")
}

#[cfg(test)]
mod tests {
    use super::{get_assets_root, get_exec_env_subdirectory, ExecutionEnvironment};
    use std::path::Path;

    #[test]
    fn test_get_exec_env_subdirectory() {
        match ExecutionEnvironment::default() {
            ExecutionEnvironment::Docker => {
                assert_eq!(get_exec_env_subdirectory(), "docker");
            }
            ExecutionEnvironment::Local => {
                assert_eq!(get_exec_env_subdirectory(), "local");
            }
        };
    }

    #[test]
    fn test_get_path_to_resources() {
        println!("{}", get_assets_root());

        assert!(Path::new(&get_assets_root()).exists());
    }
}
