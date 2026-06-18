use crate::{
    execution::hawk_main::BothEyes, hnsw::graph::GraphMutation,
    utils::serialization::types::graph_mutation_v0::GraphMutationV0,
};
use eyre::Result;
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

/* ------------- Private Helpers ------------- */

fn deserialize_v0_to_current(bytes: &[u8]) -> Result<BothEyes<Vec<GraphMutation>>> {
    let v0: BothEyes<Vec<GraphMutationV0>> = bincode::deserialize(bytes)?;
}
