use crate::types::IrisCodeBase64;
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

/// Writes an ndjson file of plaintext Iris codes.
pub fn write_plaintext_store(
    vector: &PlaintextStore,
    path_to_ndjson: &Path,
) -> std::io::Result<()> {
    // Set serial identifiers.
    let serial_ids: Vec<_> = vector.storage.get_sorted_serial_ids();

    // Write Iris codes only - ensures files are backwards compatible.
    let handle = File::create(path_to_ndjson)?;
    let mut writer = BufWriter::new(handle);
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
