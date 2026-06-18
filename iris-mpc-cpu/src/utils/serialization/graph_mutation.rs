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
    /// Designated current stable format — resolves to `V0`.
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
            GraphMutationFormat::Current | GraphMutationFormat::V0 => 0,
        }
    }
}

impl TryFrom<i16> for GraphMutationFormat {
    type Error = eyre::Error;

    fn try_from(v: i16) -> Result<Self> {
        match v {
            0 => Ok(GraphMutationFormat::V0),
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
    // V0: plain bincode — no prefix bytes in the blob itself; version is tracked
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
        // When a V1 is added: add arm here, define a types/graph_mutation_v1.rs
        // intermediate struct with From<GraphMutationV1> → BothEyes<Vec<GraphMutation>>,
        // and call bincode::deserialize::<GraphMutationV1>(bytes)?.into()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_v0_deserialize() {
        // Input in the V0 wire types: one mutation on the first eye, none on the
        // second.
        let vid1 = graph_mutation_v0::VectorId { id: 42, version: 1 };
        let vid2 = graph_mutation_v0::VectorId { id: 99, version: 2 };
        let vid3 = graph_mutation_v0::VectorId {
            id: 100,
            version: 3,
        };

        let both_eyes: BothEyes<Vec<graph_mutation_v0::GraphMutationV0>> = [
            vec![graph_mutation_v0::GraphMutationV0 {
                seq_no: 12345,
                ops: vec![
                    graph_mutation_v0::MutationOp::AddNode {
                        id: vid1.clone(),
                        height: 3,
                        update_ep: graph_mutation_v0::UpdateEntryPoint::Append { layer: 2 },
                    },
                    graph_mutation_v0::MutationOp::AddEdges {
                        base: vid2.clone(),
                        neighbors: vec![vid1.clone(), vid3.clone()],
                        layer: 1,
                        edge_type: graph_mutation_v0::EdgeType::All,
                    },
                    graph_mutation_v0::MutationOp::RemoveNode { id: vid3.clone() },
                ],
            }],
            vec![],
        ];

        // Hand-built target in the *current* types. Deliberately constructed
        // literally rather than via `.into()`, so the V0 -> current conversion is
        // actually under test instead of being asserted against itself.
        let expected: BothEyes<Vec<GraphMutation>> = [
            vec![GraphMutation {
                seq_no: 12345,
                ops: vec![
                    MutationOp::AddNode {
                        id: VectorId::new(42, 1),
                        height: 3,
                        update_ep: UpdateEntryPoint::Append { layer: 2 },
                    },
                    MutationOp::AddEdges {
                        base: VectorId::new(99, 2),
                        neighbors: vec![VectorId::new(42, 1), VectorId::new(100, 3)],
                        layer: 1,
                        edge_type: EdgeType::All,
                    },
                    MutationOp::RemoveNode {
                        id: VectorId::new(100, 3),
                    },
                ],
            }],
            vec![],
        ];

        let serialized = bincode::serialize(&both_eyes).expect("serialization failed");
        let deserialized = deserialize_mutations(GraphMutationFormat::V0, &serialized)
            .expect("deserialization failed");

        assert_eq!(deserialized, expected);
    }
}
