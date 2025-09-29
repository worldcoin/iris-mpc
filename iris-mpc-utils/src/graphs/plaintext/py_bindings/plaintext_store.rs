use crate::{misc::limited_iterator, types::IrisCodeBase64};
use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    sync::Arc,
};

pub fn from_ndjson_file(filename: &str, len: Option<usize>) -> io::Result<PlaintextStore> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    // Create an iterator over deserialized objects
    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<IrisCodeBase64>();
    let stream = limited_iterator(stream, len);

    // Iterate over each deserialized object
    let mut vector = PlaintextStore::new();
    for (idx, json_pt) in stream.into_iter().enumerate() {
        let json_pt = json_pt?;
        let iris = (&json_pt).into();
        let id = IrisVectorId::from_0_index(idx as u32);
        vector.insert_with_id(id, Arc::new(iris));
    }

    if let Some(num) = len {
        if vector.len() != num {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File {} contains too few entries; number read: {}",
                    filename,
                    vector.len()
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
    // Collect and sort keys
    let serial_ids: Vec<_> = vector.storage.get_sorted_serial_ids();
    // to keep all old ndjson files backwards compatible, we write the iris codes only
    for serial_id in serial_ids {
        let pt = vector
            .storage
            .get_vector_by_serial_id(serial_id)
            .expect("key not found in store");
        let json_pt: IrisCodeBase64 = (&**pt).into();
        serde_json::to_writer(&mut writer, &json_pt)?;
        writer.write_all(b"\n")?; // Write a newline after each JSON object
    }
    writer.flush()?;
    Ok(())
}
