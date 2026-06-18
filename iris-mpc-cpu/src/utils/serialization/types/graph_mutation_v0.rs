use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct GraphMutationV0 {
    pub seq_no: u64,
    pub ops: Vec<MutationOp>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct VectorId {
    id: u32,
    version: i16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum EdgeType {
    /// Affects forward edges from the base node
    Base,

    /// Affects back edges into the base node
    Neighbors,

    /// Affects both forward edges from and back edges into the base node
    All,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum UpdateEntryPoint {
    /// Do not update entry points based on inserted vector.
    False,

    /// Set a new unique entry point.
    SetUnique { layer: usize },

    /// Append a new entry point to the current list.
    Append { layer: usize },
}

#[derive(Clone, Serialize, Deserialize)]
enum MutationOp {
    AddNode {
        id: VectorId,
        /// Number of real graph layers this node is included in. The node will
        /// be present in layers `0..height`.
        height: usize,
        update_ep: UpdateEntryPoint,
    },
    RemoveNode {
        id: VectorId,
    },
    AddEdges {
        base: VectorId,
        neighbors: Vec<VectorId>,
        layer: usize,
        edge_type: EdgeType,
    },
    RemoveEdges {
        base: VectorId,
        neighbors: Vec<VectorId>,
        layer: usize,
        edge_type: EdgeType,
    },
}
