use crate::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_common::{
    iris_db::iris::{IrisCode, IrisCodeArray},
    IrisVectorId,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    sync::Arc,
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

/// Read `len` iris codes from the specified file into an in-memory `Vec`, or
/// all iris codes of `len` is `None`.
pub fn irises_from_ndjson_file(filename: &str, len: Option<usize>) -> io::Result<Vec<IrisCode>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Create an iterator over deserialized objects
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let stream = super::limited_iterator(stream, len);

    // Read iris codes into memory
    let codes: Vec<_> = stream
        .into_iter()
        .map(|json_ptxt| (&json_ptxt.unwrap()).into())
        .collect();

    // Check that enough codes were present in file
    if let Some(num) = len {
        if codes.len() != num {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File {} contains too few entries; number read: {}",
                    filename,
                    codes.len()
                ),
            ));
        }
    }

    Ok(codes)
}

pub fn from_ndjson_file(filename: &str, len: Option<usize>) -> io::Result<PlaintextStore> {
    let codes = irises_from_ndjson_file(filename, len)?;

    // Iterate over each deserialized object
    let mut vector = PlaintextStore::new();
    for (idx, iris) in codes.into_iter().enumerate() {
        let id = IrisVectorId::from_0_index(idx as u32);
        vector.insert_with_id(id, Arc::new(iris));
    }

    Ok(vector)
}

pub fn to_ndjson_file(vector: &PlaintextStore, filename: &str) -> std::io::Result<()> {
    // Serialize the objects to the file
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    // Collect and sort keys
    let serial_ids: Vec<_> = vector.storage.get_sorted_serial_ids();
    // to keep all old ndjson files backwards compatible, we write the iris codes only
    for serial_id in serial_ids {
        let pt = vector
            .storage
            .get_vector_by_serial_id(serial_id)
            .expect("key not found in store");
        let json_pt: Base64IrisCode = (&**pt).into();
        serde_json::to_writer(&mut writer, &json_pt)?;
        writer.write_all(b"\n")?; // Write a newline after each JSON object
    }
    writer.flush()?;
    Ok(())
}
