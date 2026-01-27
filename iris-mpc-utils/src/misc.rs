use std::{
    fs::File,
    hash::{DefaultHasher, Hash, Hasher},
    io::{BufReader, BufWriter},
    path::Path,
};

use base64::prelude::{Engine, BASE64_STANDARD};
use bincode;
use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};
use serde_json;

/// Returns a hash computed over a type instance using `DefaultHasher`.
pub fn compute_default_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Returns a boxed iterator over the first `limit` elements of `iter`.
pub fn limited_iterator<I>(iter: I, limit: Option<usize>) -> Box<dyn Iterator<Item = I::Item>>
where
    I: Iterator + 'static,
{
    match limit {
        Some(num) => Box::new(iter.take(num)),
        None => Box::new(iter),
    }
}

/// Decoder: Base64 -> T.
pub fn decode_b64<T: DeserializeOwned>(encoded: &str) -> Result<T> {
    let decoded_bytes = BASE64_STANDARD.decode(encoded)?;

    Ok(bincode::deserialize(&decoded_bytes)?)
}

/// Encoder: T -> Base64.
pub fn encode_b64<T: Serialize>(entity: &T) -> String {
    let encoded_bytes = bincode::serialize(&entity).expect("to serialize");

    BASE64_STANDARD.encode::<Vec<u8>>(encoded_bytes)
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
