use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An in-memory implementation of an HNSW hierarchical graph.
///
/// This type is a serialization-focused adapter, provided for long-term
/// compatibility and portability of serialized data.
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
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Layer {
    pub links: HashMap<VectorId, EdgeIds>,
    pub set_hash: u64,
}

/// Type associated with the `GraphV3` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeIds(pub Vec<VectorId>);

/// Type associated with the `GraphV3` serialization type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct VectorId {
    pub id: u32,
    pub version: i16,
}

/* ------------------------------- I/O ------------------------------ */

pub fn read_graph_v3<R: std::io::Read>(reader: &mut R) -> eyre::Result<GraphV3> {
    let data = bincode::deserialize_from(reader)?;
    Ok(data)
}

/// Serializes a `GraphV3` with links sorted by `(id, version)` so that all
/// parties holding identical logical graphs produce identical bytes.
///
/// Bincode encodes `HashMap<K, V>` and `Vec<(K, V)>` with the same wire
/// format (u64 length + sequential entries), so the output is fully
/// compatible with `read_graph_v3`.
pub fn write_graph_v3<W: std::io::Write>(writer: &mut W, data: &GraphV3) -> eyre::Result<()> {
    // Proxy types that serialize identically to GraphV3 / Layer / EdgeIds
    // but with sorted links so bincode output is deterministic.
    #[derive(Serialize)]
    struct SortedGraphV3<'a> {
        entry_point: &'a Vec<EntryPoint>,
        layers: Vec<SortedLayer<'a>>,
    }
    #[derive(Serialize)]
    struct SortedLayer<'a> {
        links: Vec<(&'a VectorId, SortedEdgeIds<'a>)>,
        set_hash: u64,
    }
    #[derive(Serialize)]
    struct SortedEdgeIds<'a>(Vec<&'a VectorId>);

    let sorted = SortedGraphV3 {
        entry_point: &data.entry_point,
        layers: data
            .layers
            .iter()
            .map(|layer| {
                let mut links: Vec<(&VectorId, &EdgeIds)> = layer.links.iter().collect();
                links.sort_by_key(|(k, _)| (k.id, k.version));
                SortedLayer {
                    links: links
                        .into_iter()
                        .map(|(k, v)| {
                            let mut nbrs: Vec<&VectorId> = v.0.iter().collect();
                            nbrs.sort_by_key(|n| (n.id, n.version));
                            (k, SortedEdgeIds(nbrs))
                        })
                        .collect(),
                    set_hash: layer.set_hash,
                }
            })
            .collect(),
    };

    bincode::serialize_into(writer, &sorted)?;
    Ok(())
}
