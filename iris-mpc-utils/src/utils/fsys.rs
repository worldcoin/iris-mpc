use bincode;
use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};
use serde_json;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

/// Returns path to root directory.
pub fn get_path_to_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).into()
}

/// Returns path to sub-directory.
pub fn get_path_to_subdir(name: &str) -> PathBuf {
    get_path_to_root().join(name)
}

/// Reads binary data from a file & deserializes a domain type.
pub fn read_bin<T: DeserializeOwned>(fpath: &Path) -> Result<T> {
    let file = File::open(fpath)?;
    let reader = BufReader::new(file);
    let data: T = bincode::deserialize_from(reader)?;

    Ok(data)
}

/// Reads JSON data from a file & deserializes a domain type.
pub fn read_json<T: DeserializeOwned>(fpath: &Path) -> Result<T> {
    let file = File::open(fpath)?;
    let reader = BufReader::new(file);
    let data: T = serde_json::from_reader(reader)?;

    Ok(data)
}

/// Writes binary data serialized from a domain type to a file.
pub fn write_bin<T: Serialize>(data: &T, fpath: &Path) -> Result<()> {
    let file = File::create(fpath)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;

    Ok(())
}

/// Writes JSON data serialized from a domain type to a file.
pub fn write_json<T: Serialize>(data: &T, fpath: &Path) -> Result<()> {
    let file = File::create(fpath)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &data)?;

    Ok(())
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
