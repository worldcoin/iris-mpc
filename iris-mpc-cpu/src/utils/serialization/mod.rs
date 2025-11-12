use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub mod graph;
pub mod iris_ndjson;
pub mod types;

pub fn write_bin<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

pub fn read_bin<T: DeserializeOwned>(filename: &str) -> Result<T> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data: T = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_json<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &data)?;
    Ok(())
}

pub fn read_json<T: DeserializeOwned>(filename: &str) -> Result<T> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let data: T = serde_json::from_reader(reader)?;
    Ok(data)
}
