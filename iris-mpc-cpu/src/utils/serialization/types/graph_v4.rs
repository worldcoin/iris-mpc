use serde::ser::{SerializeMap, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
///
/// `Serialize` is hand-written to emit `links` in sorted key order, mirroring
/// `layered_graph::Layer`. `HashMap`'s derived serialization is iteration-order
/// dependent, which would make checkpoint bytes (and their BLAKE3) nondeterministic
/// across processes and parties, breaking cross-party checkpoint consensus.
#[derive(Default, Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Layer {
    pub links: HashMap<VectorId, EdgeIds>,
    pub set_hash: u64,
}

struct SortedLinks<'a> {
    links: &'a HashMap<VectorId, EdgeIds>,
}

impl Serialize for SortedLinks<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entries: Vec<_> = self.links.iter().collect();
        entries.sort_by_key(|(key, _)| **key);

        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (key, value) in entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl Serialize for Layer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("Layer", 2)?;
        state.serialize_field("links", &SortedLinks { links: &self.links })?;
        state.serialize_field("set_hash", &self.set_hash)?;
        state.end()
    }
}

/// Type associated with the `GraphV4` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<VectorId>);

/// Type associated with the `GraphV4` serialization type.
#[derive(
    Copy, Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash,
)]
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
