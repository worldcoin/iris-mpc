use eyre::{bail, Result};
use serde::Deserialize;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub mod graph;
pub mod int4_ndjson;
pub mod iris_ndjson;
pub mod types;

pub fn write_bin<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    // nosemgrep: tainted-path
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

pub fn read_bin<T: DeserializeOwned>(filename: &str) -> Result<T> {
    // nosemgrep: tainted-path
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data: T = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_json<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    // nosemgrep: tainted-path
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &data)?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(filename: &str) -> Result<T> {
    // nosemgrep: tainted-path
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data: T = serde_json::from_reader(reader)?;
    Ok(data)
}

pub fn load_toml<'a, T, P>(path: P) -> Result<T>
where
    T: Deserialize<'a>,
    P: AsRef<Path>,
{
    let text = std::fs::read_to_string(path)?;
    let de = toml::de::Deserializer::new(&text);
    let t = serde_path_to_error::deserialize(de)?;
    Ok(t)
}

/// Reject a TOML config that mixes iris-store keys with deep-ID-store keys.
///
/// Binaries that dispatch on `#[serde(untagged)]` would otherwise silently pick
/// the first matching variant; this fails loudly so the operator notices.
pub fn check_store_kind_unambiguous(text: &str) -> Result<()> {
    let val: toml::Value = toml::from_str(text)?;
    let table = match val.as_table() {
        Some(t) => t,
        None => return Ok(()),
    };
    const IRIS_KEYS: &[&str] = &["irises", "irises_path", "distance_fn"];
    const DEEPID_KEYS: &[&str] = &["vectors", "vectors_path", "threshold"];
    let iris_present: Vec<&str> = IRIS_KEYS
        .iter()
        .copied()
        .filter(|k| table.contains_key(*k))
        .collect();
    let deepid_present: Vec<&str> = DEEPID_KEYS
        .iter()
        .copied()
        .filter(|k| table.contains_key(*k))
        .collect();
    if !iris_present.is_empty() && !deepid_present.is_empty() {
        bail!(
            "config mixes iris keys ({iris:?}) with deep-ID keys ({deepid:?}); \
             specify exactly one store kind",
            iris = iris_present,
            deepid = deepid_present,
        );
    }
    Ok(())
}

/// Load a TOML config, after first verifying that the file does not mix
/// iris-store and deep-ID-store top-level keys.
pub fn load_store_kind_toml<T, P>(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let text = std::fs::read_to_string(path)?;
    check_store_kind_unambiguous(&text)?;
    let de = toml::de::Deserializer::new(&text);
    let t = serde_path_to_error::deserialize(de)?;
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::check_store_kind_unambiguous;

    #[test]
    fn rejects_mixed_iris_and_deepid_keys() {
        let toml_str = r#"
distance_fn = "Simple"
threshold = 5000

[irises]
option = "Random"
number = 16

[vectors]
option = "Random"
number = 16
"#;
        let err = check_store_kind_unambiguous(toml_str).unwrap_err();
        assert!(err.to_string().contains("mixes iris keys"));
    }

    #[test]
    fn accepts_iris_only() {
        let toml_str = r#"
distance_fn = "Simple"

[irises]
option = "Random"
number = 16
"#;
        check_store_kind_unambiguous(toml_str).unwrap();
    }

    #[test]
    fn accepts_deepid_only() {
        let toml_str = r#"
threshold = 5000

[vectors]
option = "Random"
number = 16
"#;
        check_store_kind_unambiguous(toml_str).unwrap();
    }
}
