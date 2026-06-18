use crate::{
    execution::hawk_main::BothEyes,
    hnsw::graph::{
        mutation::{EdgeType, MutationOp, UpdateEntryPoint},
        GraphMutation,
    },
    utils::serialization::types::graph_mutation_v0::{self, GraphMutationV0},
};
use eyre::Result;
use iris_mpc_common::VectorId;
use serde::{Deserialize, Serialize};

/* ----------- GraphMutationFormat enum (mirrors GraphFormat) ----------- */

/// Serialization format version for `BothEyes<Vec<GraphMutation>>` blobs
/// stored in the `hawk_graph_mutations.serialized_mutations` column.
///
/// The active version is persisted in `hawk_graph_mutations.mutation_format_version`
/// so that the deserializer can dispatch without inspecting the blob bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutationFormat {
    /// Designated current stable format — resolves to `V1`.
    Current,

    /// V0: plain `bincode::serialize(BothEyes<Vec<GraphMutation>>)`.
    ///
    /// This is the format that existed before versioning was introduced.
    /// All previously-written rows are implicitly V0.
    V0,
}

impl GraphMutationFormat {
    /// Integer stored in `mutation_format_version` DB column.
    pub fn version(&self) -> i16 {
        match self {
            GraphMutationFormat::Current | GraphMutationFormat::V0 => 1,
        }
    }
}

impl TryFrom<i16> for GraphMutationFormat {
    type Error = eyre::Error;

    fn try_from(v: i16) -> Result<Self> {
        match v {
            1 => Ok(GraphMutationFormat::V0),
            _ => Err(eyre::eyre!(
                "unsupported GraphMutation format version: {}",
                v
            )),
        }
    }
}

/* ------------- Public serialization API ------------- */

/// Serialize using the current stable format.
pub fn serialize_mutations_current(data: &BothEyes<Vec<GraphMutation>>) -> Result<Vec<u8>> {
    // V1: plain bincode — no prefix bytes in the blob itself; version is tracked
    // in the DB column `mutation_format_version`.
    Ok(bincode::serialize(data)?)
}

/// Deserialize from `bytes` using the specified `format`.
///
/// `bytes` is the raw `serialized_mutations` blob from Postgres.
/// The caller is responsible for reading `mutation_format_version` from the
/// same row and converting it via `GraphMutationFormat::try_from`.
pub fn deserialize_mutations(
    format: GraphMutationFormat,
    bytes: &[u8],
) -> Result<BothEyes<Vec<GraphMutation>>> {
    match format {
        GraphMutationFormat::Current | GraphMutationFormat::V0 => deserialize_v0_to_current(bytes),
        // When a V2 is added: add arm here, define a types/graph_mutation_v2.rs
        // intermediate struct with From<GraphMutationV2> → BothEyes<Vec<GraphMutation>>,
        // and call bincode::deserialize::<GraphMutationV2>(bytes)?.into()
    }
}

/* ------------- Deserialize Helpers ------------- */

fn deserialize_v0_to_current(bytes: &[u8]) -> Result<BothEyes<Vec<GraphMutation>>> {
    let v0: BothEyes<Vec<GraphMutationV0>> = bincode::deserialize(bytes)?;
    Ok(v0.map(|eye| eye.into_iter().map(|m| m.into()).collect()))
}

/* ----------- Conversion GraphMutationV0 -> GraphMutation ----------- */

impl From<graph_mutation_v0::VectorId> for VectorId {
    fn from(value: graph_mutation_v0::VectorId) -> Self {
        VectorId::new(value.id, value.version)
    }
}

impl From<graph_mutation_v0::EdgeType> for EdgeType {
    fn from(value: graph_mutation_v0::EdgeType) -> Self {
        match value {
            graph_mutation_v0::EdgeType::Base => EdgeType::Base,
            graph_mutation_v0::EdgeType::Neighbors => EdgeType::Neighbors,
            graph_mutation_v0::EdgeType::All => EdgeType::All,
        }
    }
}

impl From<graph_mutation_v0::UpdateEntryPoint> for UpdateEntryPoint {
    fn from(value: graph_mutation_v0::UpdateEntryPoint) -> Self {
        match value {
            graph_mutation_v0::UpdateEntryPoint::False => UpdateEntryPoint::False,
            graph_mutation_v0::UpdateEntryPoint::Append { layer } => {
                UpdateEntryPoint::Append { layer }
            }
        }
    }
}

impl From<graph_mutation_v0::MutationOp> for MutationOp {
    fn from(value: graph_mutation_v0::MutationOp) -> Self {
        match value {
            graph_mutation_v0::MutationOp::AddNode {
                id,
                height,
                update_ep,
            } => MutationOp::AddNode {
                id: id.into(),
                height,
                update_ep: update_ep.into(),
            },
            graph_mutation_v0::MutationOp::RemoveNode { id } => {
                MutationOp::RemoveNode { id: id.into() }
            }
            graph_mutation_v0::MutationOp::AddEdges {
                base,
                neighbors,
                layer,
                edge_type,
            } => MutationOp::AddEdges {
                base: base.into(),
                neighbors: neighbors.into_iter().map(|n| n.into()).collect(),
                layer,
                edge_type: edge_type.into(),
            },
            graph_mutation_v0::MutationOp::RemoveEdges {
                base,
                neighbors,
                layer,
                edge_type,
            } => MutationOp::RemoveEdges {
                base: base.into(),
                neighbors: neighbors.into_iter().map(|n| n.into()).collect(),
                layer,
                edge_type: edge_type.into(),
            },
        }
    }
}

impl From<GraphMutationV0> for GraphMutation {
    fn from(value: GraphMutationV0) -> Self {
        GraphMutation {
            seq_no: value.seq_no,
            ops: value.ops.into_iter().map(|op| op.into()).collect(),
        }
    }
}
