use std::path::{Path, PathBuf};

/// Returns path to root directory.
pub fn get_path_to_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).into()
}

/// Returns path to sub-directory.
pub fn get_path_to_subdir(name: &str) -> PathBuf {
    get_path_to_root().join(name)
}

#[cfg(test)]
mod tests {
    use super::{get_path_to_root, get_path_to_subdir};

    #[test]
    fn test_get_path_to_root() {
        assert!(get_path_to_root().exists());
    }

    #[test]
    fn test_get_path_to_subdir() {
        assert!(get_path_to_subdir("assets").exists());
    }
}
