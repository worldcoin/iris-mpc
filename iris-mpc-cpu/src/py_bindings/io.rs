use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::GraphMem};
use bincode;
use csv::Writer;
use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::json;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

pub fn write_bin<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

pub fn save_excel<T: Serialize>(data: &T, filename: &str) -> Result<()> {
    let file = File::create(filename)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, data)?;
    Ok(())
}

pub fn save_graph_as_csv(
    graph: &GraphMem<PlaintextStore>,
    graph_id: usize,
    file_name: &str,
) -> eyre::Result<()> {
    let mut wtr = Writer::from_path(file_name)?;

    // Write header
    wtr.write_record(&["graph_id", "source_ref", "layer", "links"])?;

    for (layer_idx, layer) in graph.layers.iter().enumerate() {
        for (vec_id, links) in &layer.links {
            // Convert links to JSON string
            let links_json =
                serde_json::to_string(&links.iter().map(|link| json!(link)).collect::<Vec<_>>())?;
            wtr.write_record(&[
                graph_id.to_string(),
                vec_id.to_string(),
                layer_idx.to_string(),
                links_json,
            ])?;
        }
    }
    wtr.flush()?;
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
