use std::{fs, io::Error, path::Path};

use toml;

/// Returns a deserialised toml configuration file.
pub fn read_toml_config<T>(path_to_config: &Path) -> Result<T, Error>
where
    T: serde::de::DeserializeOwned,
{
    if !path_to_config.exists() {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            "Configuration file not found",
        ));
    }

    let content = fs::read_to_string(path_to_config)?;
    toml::from_str(&content).map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e))
}
