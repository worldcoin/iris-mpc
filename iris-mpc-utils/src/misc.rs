use bincode;
use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};
use serde_json;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

/// Returns a message for logging.
fn get_log_message(component: &str, msg: String) -> String {
    format!("HNSW-UTILS :: {} :: {}", component, msg)
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

/// Logs & returns a component error message.
pub(crate) fn log_error(component: &str, msg: &str) -> String {
    let msg = get_log_message(component, msg.to_string());

    // In testing print to stdout.
    #[cfg(test)]
    println!("ERROR :: {}", msg);

    // Trace as normal.
    tracing::error!(msg);

    msg
}

/// Logs & returns a component information message.
pub(crate) fn log_info(component: &str, msg: &str) -> String {
    let msg = get_log_message(component, msg.to_string());

    // In testing print to stdout.
    #[cfg(test)]
    println!("{}", msg);

    // Trace as normal.
    tracing::info!(msg);

    msg
}

/// Logs & returns a component warning message.
#[allow(dead_code)]
pub(crate) fn log_warn(component: &str, msg: &str) -> String {
    let msg = get_log_message(component, msg.to_string());

    // In testing print to stdout.
    #[cfg(test)]
    println!("WARN :: {}", msg);

    tracing::warn!(msg);

    msg
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
