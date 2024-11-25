use crate::hawkers::plaintext_store::{PlaintextIris, PlaintextPoint, PlaintextStore};
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
};

/// Iris code representation using base64 encoding compatible with Open IRIS
#[derive(Serialize, Deserialize)]
pub struct Base64IrisCode {
    iris_codes: String,
    mask_codes: String,
}

impl From<&IrisCode> for Base64IrisCode {
    fn from(value: &IrisCode) -> Self {
        Self {
            iris_codes: value.code.to_base64().unwrap(),
            mask_codes: value.mask.to_base64().unwrap(),
        }
    }
}

impl From<&Base64IrisCode> for IrisCode {
    fn from(value: &Base64IrisCode) -> Self {
        Self {
            code: IrisCodeArray::from_base64(&value.iris_codes).unwrap(),
            mask: IrisCodeArray::from_base64(&value.mask_codes).unwrap(),
        }
    }
}

pub fn from_ndjson_file(filename: &str, len: Option<usize>) -> io::Result<PlaintextStore> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Create an iterator over deserialized objects
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let stream = super::limited_iterator(stream, len);

    // Iterate over each deserialized object
    let mut vector = PlaintextStore::default();
    for json_pt in stream {
        let json_pt = json_pt?;
        vector.points.push(PlaintextPoint {
            data:          PlaintextIris((&json_pt).into()),
            is_persistent: true,
        });
    }

    if let Some(num) = len {
        if vector.points.len() != num {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File {} contains too few entries; number read: {}",
                    filename,
                    vector.points.len()
                ),
            ));
        }
    }

    Ok(vector)
}

pub fn to_ndjson_file(vector: &PlaintextStore, filename: &str) -> std::io::Result<()> {
    // Serialize the objects to the file
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    for pt in &vector.points {
        let json_pt: Base64IrisCode = (&pt.data.0).into();
        serde_json::to_writer(&mut writer, &json_pt)?;
        writer.write_all(b"\n")?; // Write a newline after each JSON object
    }
    writer.flush()?;
    Ok(())
}
