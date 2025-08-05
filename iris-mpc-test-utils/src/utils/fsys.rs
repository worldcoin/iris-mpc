use super::types::ExecutionEnvironment;

/// Returns subdirectory name for current test run environment.
pub fn get_exec_env_subdirectory() -> &'static str {
    match ExecutionEnvironment::new() {
        ExecutionEnvironment::Docker => "docker",
        ExecutionEnvironment::Local => "local",
    }
}

/// Returns path to resources assets directory.
pub fn get_assets_root() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/assets")
}

#[cfg(test)]
mod tests {
    use super::{get_assets_root, get_exec_env_subdirectory, ExecutionEnvironment};
    use std::path::Path;

    #[test]
    fn test_get_exec_env_subdirectory() {
        match ExecutionEnvironment::new() {
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
