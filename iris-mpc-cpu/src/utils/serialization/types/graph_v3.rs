use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
///
/// **Important:** This type uses deterministic serialization to ensure binary
/// equivalence of serialized graph outputs. This is required for validating
/// checkpoint equivalence between MPC nodes.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphV3 {
    pub entry_point: Vec<EntryPoint>,
    pub layers: Vec<Layer>,
}

/// Type associated with the `GraphV3` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    pub point: VectorId,
    pub layer: usize,
}

/// Type associated with the `GraphV3` serialization type.
///
/// Uses deterministic serialization by sorting HashMap entries before serialization.
/// This ensures binary equivalence for checkpoint validation across MPC nodes.
#[derive(Default, Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Layer {
    pub links: HashMap<VectorId, EdgeIds>,
    pub set_hash: u64,
}

impl Serialize for Layer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Sort HashMap entries by key for deterministic serialization
        let mut entries: Vec<_> = self.links.iter().collect();
        entries.sort_by(|(left, _), (right, _)| left.cmp(right));

        // Serialize as a map within the struct
        let mut state = serializer.serialize_struct("Layer", 2)?;

        // Serialize sorted links as a sequence of key-value pairs
        // This matches bincode's map serialization format
        state.serialize_field("links", &SortedLinks { entries: &entries })?;
        state.serialize_field("set_hash", &self.set_hash)?;
        state.end()
    }
}

/// Helper struct for serializing sorted HashMap entries.
struct SortedLinks<'a> {
    entries: &'a [(&'a VectorId, &'a EdgeIds)],
}

impl<'a> Serialize for SortedLinks<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (key, value) in self.entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

/// Type associated with the `GraphV3` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<VectorId>);

/// Type associated with the `GraphV3` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
pub struct VectorId {
    pub id: u32,
    pub version: i16,
}

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v3<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV3> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v3<W: std::io::Write>(writer: &mut W, data: &GraphV3) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
