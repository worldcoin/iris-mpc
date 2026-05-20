use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
#[derive(Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct GraphV4 {
    pub entry_points: Vec<EntryPoint>,
    pub layers: Vec<Layer>,
    pub last_update_seq_no: u64,
}

/// Type associated with the `GraphV4` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    pub point: VectorId,
    pub layer: usize,
}

/// Type associated with the `GraphV4` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    pub links: HashMap<VectorId, EdgeIds>,
    pub set_hash: u64,
}

/// Type associated with the `GraphV4` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<VectorId>);

/// Type associated with the `GraphV4` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct VectorId {
    pub id: u32,
    pub version: i16,
}

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v4<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV4> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v4<W: std::io::Write>(writer: &mut W, data: &GraphV4) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
