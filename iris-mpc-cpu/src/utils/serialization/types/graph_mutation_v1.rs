use serde::{Deserialize, Serialize};

/// V1 wire format for one WAL graph mutation.
///
/// Differences from V0: edge ops carry bare serial ids — node identity/version
/// rides only `AddNode`/`RemoveNode` — and ops record the intent verbatim,
/// anchored by `as_of`; every apply resolves the references against it
/// (staleness cleanup likewise re-derives on replay). V0 is not readable; the
/// WAL is reset at the v5 cutover.
#[derive(Clone, Serialize, Deserialize)]
pub struct GraphMutationV1 {
    pub seq_no: u64,
    pub as_of: u64,
    pub ops: Vec<MutationOp>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorId {
    pub id: u32,
    pub version: i16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EdgeType {
    /// Affects forward edges from the base node
    Base,

    /// Affects back edges into the base node
    Neighbors,

    /// Affects both forward edges from and back edges into the base node
    All,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum UpdateEntryPoint {
    /// Do not update entry points based on inserted vector.
    False,

    /// Append a new entry point to the current list.
    Append { layer: usize },
}

#[derive(Clone, Serialize, Deserialize)]
pub enum MutationOp {
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
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
        edge_type: EdgeType,
    },
    RemoveEdges {
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
        edge_type: EdgeType,
    },
}
