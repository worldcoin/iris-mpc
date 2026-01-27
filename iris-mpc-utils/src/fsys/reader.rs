use std::{
    fs::{self, File},
    io::{BufReader, Error},
    path::Path,
};

use toml;

/// Returns a deserialised toml file.
pub fn read_toml<T>(path_to_file: &Path) -> Result<T, Error>
where
    T: serde::de::DeserializeOwned,
{
    if !path_to_file.exists() {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            "Toml file not found",
        ));
    }

    let content = fs::read_to_string(path_to_file)?;

    toml::from_str(&content).map_err(|e| Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Returns an iterable dataset from a json file.
pub fn read_json_iter<T>(
    path_to_file: &Path,
    n_to_skip: usize,
    n_to_take: usize,
) -> Result<impl Iterator<Item = Result<T, serde_json::Error>>, Error>
where
    T: serde::de::DeserializeOwned,
{
    if !path_to_file.exists() {
        return Err(Error::new(
            std::io::ErrorKind::NotFound,
            "JSON file not found",
        ));
    }

    let handle = File::open(path_to_file)?;
    let reader = BufReader::new(handle);
    let iterable = serde_json::Deserializer::from_reader(reader)
        .into_iter::<T>()
        .skip(n_to_skip)
        .take(n_to_take);

    Ok(iterable)
}
