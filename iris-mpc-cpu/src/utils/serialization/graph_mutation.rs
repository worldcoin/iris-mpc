use crate::{
    execution::hawk_main::BothEyes,
    hnsw::graph::{
        mutation::{EdgeType, MutationOp, UpdateEntryPoint},
        GraphMutation,
    },
    utils::serialization::types::graph_mutation_v1::{self, GraphMutationV1},
};
use eyre::{Result, WrapErr};
use iris_mpc_common::VectorId;
use serde::{Deserialize, Serialize};

/* ----------- GraphMutationFormat enum (mirrors GraphFormat) ----------- */

/// Serialization format version for `BothEyes<Vec<GraphMutation>>` blobs
/// stored in the `hawk_graph_mutations.serialized_mutations` column.
///
/// The active version is persisted in `hawk_graph_mutations.mutation_format_version`
/// so that the deserializer can dispatch without inspecting the blob bytes.
///
/// A format version pins *replay behavior*, not just the byte layout: replay
/// re-runs the same apply as minting, staleness filter included, so changing
/// the filter predicate or the filter-on-bump discipline changes the physical
/// state a recorded stream replays to, which the consensus checksum compares.
/// The hard requirement for such changes is a checkpoint barrier (no
/// pre-change segments left to replay); the version bump is the tripwire that
/// makes a stray old segment fail loud instead of replaying into
/// checksum-divergent state.
///
/// V0 (edge ops carrying `VectorId`) is intentionally not readable: the WAL is
/// reset at the v5 cutover, so a version-0 row reaching this code is an
/// operational error and fails loudly in `try_from`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutationFormat {
    /// V1: plain `bincode::serialize(BothEyes<Vec<GraphMutation>>)` with edge
    /// ops carrying bare serial ids; ops record intent only (staleness cleanup
    /// re-derives on replay).
    V1,
}

impl GraphMutationFormat {
    /// The format new writes are emitted in.
    pub const CURRENT: Self = Self::V1;

    /// Integer stored in `mutation_format_version` DB column.
    pub fn version(&self) -> i16 {
        match self {
            GraphMutationFormat::V1 => 1,
        }
    }
}

impl TryFrom<i16> for GraphMutationFormat {
    type Error = eyre::Error;

    fn try_from(v: i16) -> Result<Self> {
        match v {
            1 => Ok(GraphMutationFormat::V1),
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
        GraphMutationFormat::V1 => deserialize_v1_to_current(bytes),
        // When a V2 is added: add arm here, define a types/graph_mutation_v2.rs
        // intermediate struct with From<GraphMutationV2> → BothEyes<Vec<GraphMutation>>,
        // and call bincode::deserialize::<GraphMutationV2>(bytes)?.into()
    }
}

/* ------------- Deserialize Helpers ------------- */

fn deserialize_v1_to_current(bytes: &[u8]) -> Result<BothEyes<Vec<GraphMutation>>> {
    let v1: BothEyes<Vec<GraphMutationV1>> = bincode::deserialize(bytes).wrap_err_with(|| {
        format!(
            "deserializing GraphMutation V1 blob ({} bytes)",
            bytes.len()
        )
    })?;
    Ok(v1.map(|eye| eye.into_iter().map(|m| m.into()).collect()))
}

/* ----------- Conversion GraphMutationV1 -> GraphMutation ----------- */

impl From<graph_mutation_v1::VectorId> for VectorId {
    fn from(value: graph_mutation_v1::VectorId) -> Self {
        VectorId::new(value.id, value.version)
    }
}

impl From<graph_mutation_v1::EdgeType> for EdgeType {
    fn from(value: graph_mutation_v1::EdgeType) -> Self {
        match value {
            graph_mutation_v1::EdgeType::Base => EdgeType::Base,
            graph_mutation_v1::EdgeType::Neighbors => EdgeType::Neighbors,
            graph_mutation_v1::EdgeType::All => EdgeType::All,
        }
    }
}

impl From<graph_mutation_v1::UpdateEntryPoint> for UpdateEntryPoint {
    fn from(value: graph_mutation_v1::UpdateEntryPoint) -> Self {
        match value {
            graph_mutation_v1::UpdateEntryPoint::False => UpdateEntryPoint::False,
            graph_mutation_v1::UpdateEntryPoint::Append { layer } => {
                UpdateEntryPoint::Append { layer }
            }
        }
    }
}

impl From<graph_mutation_v1::MutationOp> for MutationOp {
    fn from(value: graph_mutation_v1::MutationOp) -> Self {
        match value {
            graph_mutation_v1::MutationOp::AddNode {
                id,
                height,
                update_ep,
            } => MutationOp::AddNode {
                id: id.into(),
                height,
                update_ep: update_ep.into(),
            },
            graph_mutation_v1::MutationOp::RemoveNode { id } => {
                MutationOp::RemoveNode { id: id.into() }
            }
            graph_mutation_v1::MutationOp::AddEdges {
                base,
                neighbors,
                layer,
                edge_type,
            } => MutationOp::AddEdges {
                base,
                neighbors,
                layer,
                edge_type: edge_type.into(),
            },
            graph_mutation_v1::MutationOp::RemoveEdges {
                base,
                neighbors,
                layer,
                edge_type,
            } => MutationOp::RemoveEdges {
                base,
                neighbors,
                layer,
                edge_type: edge_type.into(),
            },
        }
    }
}

impl From<GraphMutationV1> for GraphMutation {
    fn from(value: GraphMutationV1) -> Self {
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
    fn validate_v1_deserialize() {
        // Input in the V1 wire types: one mutation on the first eye, none on the
        // second.
        let vid1 = graph_mutation_v1::VectorId { id: 42, version: 1 };
        let vid3 = graph_mutation_v1::VectorId {
            id: 100,
            version: 3,
        };

        let both_eyes: BothEyes<Vec<graph_mutation_v1::GraphMutationV1>> = [
            vec![graph_mutation_v1::GraphMutationV1 {
                seq_no: 12345,
                ops: vec![
                    graph_mutation_v1::MutationOp::AddNode {
                        id: vid1.clone(),
                        height: 3,
                        update_ep: graph_mutation_v1::UpdateEntryPoint::Append { layer: 2 },
                    },
                    graph_mutation_v1::MutationOp::AddEdges {
                        base: 99,
                        neighbors: vec![42, 100],
                        layer: 1,
                        edge_type: graph_mutation_v1::EdgeType::All,
                    },
                    graph_mutation_v1::MutationOp::RemoveEdges {
                        base: 42,
                        neighbors: vec![100],
                        layer: 0,
                        edge_type: graph_mutation_v1::EdgeType::Base,
                    },
                    graph_mutation_v1::MutationOp::RemoveNode { id: vid3.clone() },
                ],
            }],
            vec![],
        ];

        // Hand-built target in the *current* types. Deliberately constructed
        // literally rather than via `.into()`, so the V1 -> current conversion is
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
                        base: 99,
                        neighbors: vec![42, 100],
                        layer: 1,
                        edge_type: EdgeType::All,
                    },
                    MutationOp::RemoveEdges {
                        base: 42,
                        neighbors: vec![100],
                        layer: 0,
                        edge_type: EdgeType::Base,
                    },
                    MutationOp::RemoveNode {
                        id: VectorId::new(100, 3),
                    },
                ],
            }],
            vec![],
        ];

        let serialized = bincode::serialize(&both_eyes).expect("serialization failed");
        let deserialized = deserialize_mutations(GraphMutationFormat::V1, &serialized)
            .expect("deserialization failed");

        assert_eq!(deserialized, expected);
    }

    /// V0 rows (edge ops carrying VectorId) must fail loudly: the WAL is
    /// reset at cutover, so one reaching this code is an operational error.
    #[test]
    fn version_0_is_rejected() {
        assert!(GraphMutationFormat::try_from(0).is_err());
    }
}
