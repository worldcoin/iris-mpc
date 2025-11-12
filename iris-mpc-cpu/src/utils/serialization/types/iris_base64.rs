//! Implements a data serialization format targeting the `IrisCode` type.
//!
//! This format is meant to be compatible with the base64 encoding used by
//! the Open IRIS Python library.

use eyre::Result;
use serde::{Deserialize, Serialize};

/// Iris code representation using base64 encoding compatible with Open IRIS.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64IrisCode {
    pub iris_codes: String,
    pub mask_codes: String,
}

/* ------------------------------- I/O ------------------------------ */

pub fn read_iris_base64<R: std::io::Read>(reader: &mut R) -> Result<Base64IrisCode> {
    let data = serde_json::from_reader(reader)?;
    Ok(data)
}

pub fn write_iris_base64<W: std::io::Write>(writer: &mut W, data: &Base64IrisCode) -> Result<()> {
    serde_json::to_writer(writer, &data)?;
    Ok(())
}

pub fn read_from_iris_ndjson<R: std::io::Read>(
    reader: R,
) -> impl Iterator<Item = Result<Base64IrisCode>> {
    let iter = serde_json::Deserializer::from_reader(reader)
        .into_iter()
        .map(|res| res.map_err(Into::into));
    iter
}

pub fn write_to_iris_ndjson<W: std::io::Write, D: std::iter::Iterator<Item = Base64IrisCode>>(
    writer: &mut W,
    data: D,
) -> Result<()> {
    for json_pt in data {
        serde_json::to_writer(&mut *writer, &json_pt)?;
        writer.write_all(b"\n")?; // Write a newline after each JSON object
    }
    Ok(())
}
