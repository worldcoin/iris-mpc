//! Implements a data serialization format targeting the `GraphMem` type.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphV1 {
    pub entry_point: Option<EntryPoint>,
    pub layers: Vec<Layer>,
}

/// Type associated with the `GraphV1` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    pub point: VectorId,
    pub layer: usize,
}

/// Type associated with the `GraphV1` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    pub links: HashMap<VectorId, EdgeIds>,
    pub set_hash: SetHash,
}

/// Type associated with the `GraphV1` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetHash {
    pub accumulator: u64,
}

/// Type associated with the `GraphV1` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<VectorId>);

/// Type associated with the `GraphV1` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct VectorId {
    pub id: u32,
    pub version: i16,
}

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v1<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV1> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v1<W: std::io::Write>(writer: &mut W, data: &GraphV1) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
