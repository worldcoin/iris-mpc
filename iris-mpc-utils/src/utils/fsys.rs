use super::types::ExecutionHost;
use std::path::{Path, PathBuf};

/// Returns name of an execution host specific assets subdirectory.
pub fn get_execution_host_subdirectory() -> &'static str {
    match ExecutionHost::default() {
        ExecutionHost::BareMetal => "baremetal",
        ExecutionHost::Docker => "docker",
    }
}

/// Returns path to an asset within assets directory.
pub fn get_path_to_asset(subpath: &str) -> PathBuf {
    let path_to_root = get_path_to_subdir("assets");

    Path::new(&path_to_root).join(subpath)
}

/// Returns path to assets directory.
pub fn get_path_to_assets_root() -> String {
    format!("{}/assets", get_path_to_root())
}

/// Returns path to a resource within data directory.
#[allow(dead_code)]
pub fn get_path_to_data(subpath: &str) -> PathBuf {
    let path_to_root = get_path_to_subdir("data");

    Path::new(&path_to_root).join(subpath)
}

/// Returns path to data directory.
pub fn get_path_to_data_root() -> String {
    format!("{}/data", get_path_to_root())
}

/// Returns path to data directory.
pub fn get_path_to_root() -> String {
    env!("CARGO_MANIFEST_DIR").to_string()
}

/// Returns path to data directory.
pub fn get_path_to_subdir(name: &str) -> String {
    format!("{}/{}", get_path_to_root(), name)
}

#[cfg(test)]
mod tests {
    use super::get_path_to_subdir;
    use std::path::Path;

    #[test]
    fn test_get_path_to_assets() {
        assert!(Path::new(&get_path_to_subdir("assets")).exists());
    }
}
