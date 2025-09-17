use std::path::{Path, PathBuf};

/// Returns path to root directory.
pub fn get_path_to_root_1() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).into()
}

/// Returns path to sub-directory.
pub fn get_path_to_subdir_1(name: &str) -> PathBuf {
    get_path_to_root_1().join(name)
}

#[cfg(test)]
mod tests {
    use super::{get_path_to_root_1, get_path_to_subdir_1};

    #[test]
    fn test_get_path_to_root() {
        assert!(get_path_to_root_1().exists());
    }

    #[test]
    fn test_get_path_to_subdir() {
        assert!(get_path_to_subdir_1("assets").exists());
    }
}
