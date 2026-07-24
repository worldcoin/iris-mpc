use crate::hnsw::graph::graph_store::GraphCheckpointRow;
use eyre::eyre;
use iris_mpc_common::SerialId;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, str::FromStr};

// types for the graph checkpoint sync
pub type Blake3Hash = [u8; 32];
pub type GraphCheckpointHashes = [Blake3Hash; 10];
pub const GRAPH_CHECKPOINT_ROUTE: &str = "/graph-checkpoint";
pub const GRAPH_CHECKPOINT_ENDPOINT: &str = "graph-checkpoint";

/// Default retention factor for the sparse tier of [`PruningMode::Tiered`]:
/// keep every 4th version once a checkpoint is old enough to be thinned.
pub const DEFAULT_TIERED_KEEP_EVERY_NTH: usize = 4;

pub const DEFAULT_TIERED_DELETE_OLDER_THAN_DAYS: usize = 60;
pub const DEFAULT_TIERED_THIN_OLDER_THAN_DAYS: usize = 30;

/// Env var holding the `delete_older_than` (`X`) bound for [`PruningMode::Tiered`].
pub const ENV_TIERED_DELETE_OLDER_THAN: &str = "PRUNING_TIERED_DELETE_OLDER_THAN";
/// Env var holding the `thin_older_than` (`Y`) bound for [`PruningMode::Tiered`].
pub const ENV_TIERED_THIN_OLDER_THAN: &str = "PRUNING_TIERED_THIN_OLDER_THAN";
/// Env var holding the `keep_every_nth` factor for [`PruningMode::Tiered`]
/// (optional; defaults to [`DEFAULT_TIERED_KEEP_EVERY_NTH`]).
pub const ENV_TIERED_KEEP_EVERY_NTH: &str = "PRUNING_TIERED_KEEP_EVERY_NTH";

/// Controls which older checkpoints are deleted during cleanup.
///
/// "Version age" is a checkpoint's rank when all checkpoints are ordered
/// newest-first (0 = the newest checkpoint, 1 = the next newest, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PruningMode {
    /// Do not prune any checkpoints.
    None,
    /// Prune older checkpoints that are not marked archival (default).
    OlderNonArchival,
    /// Prune all older checkpoints regardless of archival flag.
    AllOlder,
    /// Tiered retention based on version age (0 = newest):
    /// - keep every version newer than `thin_older_than` versions (recent tier),
    /// - keep only every `keep_every_nth`-th version between `thin_older_than`
    ///   and `delete_older_than` versions (sparse tier),
    /// - delete every version older than `delete_older_than` versions.
    ///
    /// The numeric bounds live in [`TieredPruningConfig`] (carried on the
    /// sidecar / genesis config), not in the variant itself.
    Tiered,
}

/// Numeric tuning knobs for [`PruningMode::Tiered`].
///
/// "Version age" is a checkpoint's rank when all checkpoints are ordered
/// newest-first (0 = the newest checkpoint, 1 = the next newest, ...).
///
/// Requires `thin_older_than <= delete_older_than` and `keep_every_nth >= 1`
/// (enforced by [`TieredPruningConfig::validate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct TieredPruningConfig {
    /// Delete all versions older than this many versions (the `X` bound).
    #[serde(default = "default_delete_older_than_days")]
    pub delete_older_than_days: usize,
    /// Start thinning versions older than this many versions (the `Y` bound).
    #[serde(default = "default_thin_older_than_days")]
    pub thin_older_than_days: usize,
    /// In the sparse tier, keep one version out of every `keep_every_nth`.
    #[serde(default = "default_keep_every_nth")]
    pub keep_every_nth: usize,
}

fn default_delete_older_than_days() -> usize {
    60
}

fn default_thin_older_than_days() -> usize {
    30
}

fn default_keep_every_nth() -> usize {
    4
}

impl TieredPruningConfig {
    /// Const-constructible default (usable in `const` items). All bounds zero
    /// except `keep_every_nth`, which uses [`DEFAULT_TIERED_KEEP_EVERY_NTH`].
    pub const DEFAULT: Self = Self {
        delete_older_than_days: DEFAULT_TIERED_DELETE_OLDER_THAN_DAYS,
        thin_older_than_days: DEFAULT_TIERED_THIN_OLDER_THAN_DAYS,
        keep_every_nth: DEFAULT_TIERED_KEEP_EVERY_NTH,
    };
}

impl Default for TieredPruningConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TieredPruningConfig {
    /// Builds a [`TieredPruningConfig`] from environment variables:
    /// - [`ENV_TIERED_DELETE_OLDER_THAN`] (`X`, required),
    /// - [`ENV_TIERED_THIN_OLDER_THAN`] (`Y`, required),
    /// - [`ENV_TIERED_KEEP_EVERY_NTH`] (`N`, optional, defaults to
    ///   [`DEFAULT_TIERED_KEEP_EVERY_NTH`]).
    ///
    /// Requires `thin_older_than <= delete_older_than` and `keep_every_nth >= 1`.
    pub fn from_env() -> Result<Self, eyre::Error> {
        let required_usize = |name: &str| -> Result<usize, eyre::Error> {
            let raw = std::env::var(name)
                .map_err(|_| eyre!("tiered pruning mode requires env var {name} to be set"))?;
            raw.parse::<usize>()
                .map_err(|e| eyre!("invalid {name} value '{raw}': {e}"))
        };

        let delete_older_than_days = required_usize(ENV_TIERED_DELETE_OLDER_THAN)?;
        let thin_older_than_days = required_usize(ENV_TIERED_THIN_OLDER_THAN)?;
        let keep_every_nth = match std::env::var(ENV_TIERED_KEEP_EVERY_NTH) {
            Ok(raw) => raw
                .parse::<usize>()
                .map_err(|e| eyre!("invalid {ENV_TIERED_KEEP_EVERY_NTH} value '{raw}': {e}"))?,
            Err(_) => DEFAULT_TIERED_KEEP_EVERY_NTH,
        };

        let cfg = Self {
            delete_older_than_days,
            thin_older_than_days,
            keep_every_nth,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validates the tiered bounds: `thin_older_than <= delete_older_than` and
    /// `keep_every_nth >= 1`.
    pub fn validate(&self) -> Result<(), eyre::Error> {
        if self.thin_older_than_days > self.delete_older_than_days {
            return Err(eyre!(
                "invalid tiered pruning config: thin_older_than_days ({}) must be \
                 <= delete_older_than_days ({})",
                self.thin_older_than_days,
                self.delete_older_than_days
            ));
        }
        if self.keep_every_nth < 1 {
            return Err(eyre!(
                "invalid tiered pruning config: keep_every_nth must be >= 1"
            ));
        }
        Ok(())
    }
}

impl Display for PruningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruningMode::None => write!(f, "none"),
            PruningMode::OlderNonArchival => write!(f, "older-non-archival"),
            PruningMode::AllOlder => write!(f, "all-older"),
            PruningMode::Tiered => write!(f, "tiered"),
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
            "tiered" => Ok(PruningMode::Tiered),
            _ => Err(eyre!(
                "invalid pruning mode: '{}', expected one of: none, older-non-archival, \
                 all-older, tiered",
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
    pub last_indexed_iris_id: SerialId,
    /// Last modification ID included in this checkpoint
    pub last_indexed_modification_id: i64,
    /// Last graph mutation ID included in this checkpoint (optional)
    pub graph_mutation_id: Option<i64>,
    /// BLAKE3 hash of the checkpoint data for integrity verification
    pub blake3_hash: String,
    /// Corresponds to the GraphFormat enum
    pub graph_version: i32,
    /// Whether this checkpoint is archival (i.e. should be retained by pruning).
    pub is_archival: bool,
}

impl GraphCheckpointState {
    /// Returns the graph_mutation_id, or an error if it is None.
    pub fn graph_mutation_id(&self) -> eyre::Result<i64> {
        self.graph_mutation_id.ok_or_else(|| {
            eyre!(
                "graph_mutation_id is not set for checkpoint: {}",
                self.s3_key
            )
        })
    }
}

impl TryFrom<GraphCheckpointRow> for GraphCheckpointState {
    type Error = eyre::Error;
    fn try_from(value: GraphCheckpointRow) -> Result<Self, Self::Error> {
        let last_indexed_iris_id: SerialId =
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
            graph_mutation_id: value.graph_mutation_id,
            blake3_hash: value.blake3_hash,
            graph_version: value.graph_version,
            is_archival: value.is_archival,
        })
    }
}
