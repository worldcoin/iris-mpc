use crate::hnsw::graph::graph_store::GraphCheckpointRow;
use eyre::eyre;
use iris_mpc_common::IrisSerialId;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr};

/// Controls which older checkpoints are deleted during cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMode {
    /// Do not prune any checkpoints.
    None,
    /// Prune older checkpoints that are not marked archival (default).
    OlderNonArchival,
    /// Prune all older checkpoints regardless of archival flag.
    AllOlder,
}

impl Display for PruningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruningMode::None => write!(f, "none"),
            PruningMode::OlderNonArchival => write!(f, "older-non-archival"),
            PruningMode::AllOlder => write!(f, "all-older"),
        }
    }
}

impl FromStr for PruningMode {
    type Err = eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(PruningMode::None),
            "older-non-archival" => Ok(PruningMode::OlderNonArchival),
            "all-older" => Ok(PruningMode::AllOlder),
            _ => Err(eyre!(
                "invalid pruning mode: '{}', expected one of: none, older-non-archival, all-older",
                s
            )),
        }
    }
}

/// Metadata stored in genesis_graph_checkpoint table for graph checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCheckpointState {
    /// S3 key where the checkpoint is stored
    pub s3_key: String,
    /// Last iris serial ID included in this checkpoint
    pub last_indexed_iris_id: IrisSerialId,
    /// Last modification ID included in this checkpoint
    pub last_indexed_modification_id: i64,
    /// BLAKE3 hash of the checkpoint data for integrity verification
    pub blake3_hash: String,
    /// Whether this checkpoint is archival (i.e. should be retained by pruning).
    pub is_archival: bool,
}

impl TryFrom<GraphCheckpointRow> for GraphCheckpointState {
    type Error = eyre::Error;
    fn try_from(value: GraphCheckpointRow) -> Result<Self, Self::Error> {
        let last_indexed_iris_id: IrisSerialId =
            value.last_indexed_iris_id.try_into().map_err(|_| {
                eyre!(
                    "Invalid last_indexed_iris_id for checkpoint: {}",
                    value.last_indexed_iris_id
                )
            })?;

        Ok(Self {
            s3_key: value.s3_key,
            last_indexed_iris_id,
            last_indexed_modification_id: value.last_indexed_modification_id,
            blake3_hash: value.blake3_hash,
            is_archival: value.is_archival,
        })
    }
}
