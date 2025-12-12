//! Implements a data serialization format targeting the `GraphMem` type
//! for a plaintext vector store.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphV0 {
    pub entry_point: Option<EntryPoint>,
    pub layers: Vec<Layer>,
}

/// Type associated with the `GraphV0` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    pub point: PointId,
    pub layer: usize,
}

/// Type associated with the `GraphV0` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    pub links: HashMap<PointId, Edges>,
}

/// Type associated with the `GraphV0` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edges(pub Vec<(PointId, (u16, u16))>);

/// Type associated with the `GraphV0` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PointId(pub u32);

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v0<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV0> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v0<W: std::io::Write>(writer: &mut W, data: &GraphV0) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
