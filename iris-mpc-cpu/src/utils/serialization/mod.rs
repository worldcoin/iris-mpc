use eyre::Result;
use serde::Deserialize;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub mod graph;
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
