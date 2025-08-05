use super::types::ExecutionEnvironment;

/// Returns subdirectory name for current test run environment.
pub fn get_subdirectory_of_exec_env() -> &'static str {
    match ExecutionEnvironment::new() {
        ExecutionEnvironment::Docker => "docker",
        ExecutionEnvironment::Local => "local",
    }
}

/// Returns path to resources assets directory.
pub fn get_path_to_assets() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/assets")
}

#[cfg(test)]
mod tests {
    use super::{get_path_to_assets, get_subdirectory_of_exec_env, ExecutionEnvironment};
    use std::path::Path;

    #[test]
    fn test_get_subdirectory_of_exec_env() {
        match ExecutionEnvironment::new() {
            ExecutionEnvironment::Docker => {
                assert_eq!(get_subdirectory_of_exec_env(), "docker");
            }
            ExecutionEnvironment::Local => {
                assert_eq!(get_subdirectory_of_exec_env(), "local");
            }
        };
    }

    #[test]
    fn test_get_path_to_resources() {
        println!("{}", get_path_to_assets());

        assert!(Path::new(&get_path_to_assets()).exists());
    }
}
