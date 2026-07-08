use eyre::{bail, Error};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Selects how the genesis delta phase reconciles the graph and HNSW iris
/// store to the source iris store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DeltaMode {
    /// Replay the source `modifications` log (default, historical behavior).
    #[default]
    Modifications,
    /// Reconcile by `(serial → version)` comparison against a pinned checkpoint.
    VersionJoin,
}

impl FromStr for DeltaMode {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "modifications" => Ok(DeltaMode::Modifications),
            "version-join" => Ok(DeltaMode::VersionJoin),
            _ => bail!(
                "invalid delta mode: '{}', expected one of: modifications, version-join",
                s
            ),
        }
    }
}
