use serde::ser::{SerializeMap, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Wire-format adapter for `GraphMem` — the `GraphV5` on-disk / checkpoint
/// serialization type, kept for long-term compatibility and portability of
/// serialized data.
///
/// `Serialize` is hand-written: both field order AND the `node_init`
/// map ordering must match `GraphMem`'s own serializer, because genesis writes
/// `[GraphMem; 2]` directly while the materializer and hawk restart read it
/// back as `GraphV5`. A derived impl would emit `node_init` in HashMap
/// iteration order, diverging from `GraphMem` and breaking checkpoint BLAKE3.
#[derive(Default, Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GraphV5 {
    pub entry_points: Vec<EntryPoint>,
    pub layers: Vec<Layer>,
    pub last_update_seq_no: u64,
    pub node_init: HashMap<SerialId, NodeInit>,
}

/// Type associated with the `GraphV5` serialization type. Wire layout must
/// match `layered_graph::NodeInit` (u64 seq_no, then i16 version).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInit {
    pub seq_no: u64,
    pub version: i16,
}

impl Serialize for GraphV5 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("GraphV5", 4)?;
        state.serialize_field("entry_points", &self.entry_points)?;
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("last_update_seq_no", &self.last_update_seq_no)?;
        let sorted_node_init: BTreeMap<_, _> = self.node_init.iter().collect();
        state.serialize_field("node_init", &sorted_node_init)?;
        state.end()
    }
}

/// Type associated with the `GraphV5` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint {
    pub point: SerialId,
    pub layer: usize,
}

/// Type associated with the `GraphV5` serialization type.
///
/// `Serialize` is hand-written to emit `links` in sorted key order, mirroring
/// `layered_graph::Layer`. `HashMap`'s derived serialization is iteration-order
/// dependent, which would make checkpoint bytes (and their BLAKE3) nondeterministic
/// across processes and parties, breaking cross-party checkpoint consensus.
#[derive(Default, Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Layer {
    pub links: HashMap<SerialId, Neighborhood>,
    pub set_hash: u64,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighborhood {
    pub neighbors: Vec<SerialId>,
    pub updated_seq_no: u64,
}

struct SortedLinks<'a> {
    links: &'a HashMap<SerialId, Neighborhood>,
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

/// Type associated with the `GraphV5` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<SerialId>);

/// Type associated with the `GraphV5` serialization type.
pub type SerialId = u32;

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v5<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV5> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

pub fn write_graph_v5<W: std::io::Write>(writer: &mut W, data: &GraphV5) -> eyre::Result<()> {
    bincode::serialize_into(writer, data)?;
    Ok(())
}
