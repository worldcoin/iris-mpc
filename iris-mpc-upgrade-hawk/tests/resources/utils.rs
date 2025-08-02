use crate::utils::TestRunExecutionEnvironment;

/// Returns subdirectory name for current test run environment.
pub(super) fn get_subdirectory_of_exec_env() -> &'static str {
    match TestRunExecutionEnvironment::new() {
        TestRunExecutionEnvironment::Docker => "docker",
        TestRunExecutionEnvironment::Local => "local",
    }
}

/// Returns path to resources assets directory.
pub(super) fn get_path_to_assets() -> String {
    let crate_root = env!("CARGO_MANIFEST_DIR");

    format!("{crate_root}/tests/assets")
}

#[cfg(test)]
mod tests {
    use super::{get_path_to_assets, get_subdirectory_of_exec_env, TestRunExecutionEnvironment};
    use std::path::Path;

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
        assert!(Path::new(&get_path_to_assets()).exists());
    }
}
