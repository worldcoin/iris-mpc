use super::BatchSizeConfig;
use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, SerialId};
use serde::{Deserialize, Serialize};

/// Genesis configuration compared for equality across nodes. This is a network
/// level type.
///
/// Only operator-intent / genuinely-global fields belong here. Party-local
/// state (persistent-state cursors, modification lists derived from them) must
/// stay out: it is exactly the divergent-but-repairable state the delta phase
/// reconciles, and whole-struct equality would wedge the run on it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    // Batch size configuration (static or dynamic with cap).
    pub batch_size_config: BatchSizeConfig,

    // Set of identifiers of Iris's to be excluded from indexation.
    pub excluded_serial_ids: Vec<SerialId>,

    // Identifier of the last Iris serial ID to be indexed.
    pub max_indexation_id: SerialId,

    // Pinned base checkpoint blake3 hash; must agree across nodes.
    pub base_checkpoint_hash: Option<String>,
}

/// Constructor.
impl Config {
    pub fn new(
        batch_size_config: BatchSizeConfig,
        excluded_serial_ids: Vec<SerialId>,
        max_indexation_id: SerialId,
        base_checkpoint_hash: Option<String>,
    ) -> Self {
        Self {
            batch_size_config,
            excluded_serial_ids,
            max_indexation_id,
            base_checkpoint_hash,
        }
    }
}

/// Encapsulates a node's synchronization state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    // Configuration common to all nodes.
    pub common_config: CommonConfig,

    // Configuration common to all nodes during genesis.
    pub genesis_config: Config,
}

/// Constructor.
impl SyncState {
    pub fn new(common_config: CommonConfig, genesis_config: Config) -> Self {
        Self {
            common_config,
            genesis_config,
        }
    }
}

/// Encapsulates a result over a node's synchronization state evaluation.
pub struct SyncResult {
    // Own synchronization state.
    pub my_state: SyncState,

    // Network synchronization state.
    pub all_states: Vec<SyncState>,
}

/// Constructor.
impl SyncResult {
    pub fn new(my_state: SyncState, all_states: Vec<SyncState>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }
}

/// Methods.
impl SyncResult {
    /// Check if the common part of the config is the same across all nodes.
    pub fn check_synced_state(&self) -> Result<()> {
        self.check_software_version()?;
        for state in &self.all_states {
            ensure!(
                *state == self.my_state,
                "Inconsistent genesis config: \nhave: {} \ngot: {}",
                summarize_sync_state(&self.my_state),
                summarize_sync_state(state)
            );
        }
        Ok(())
    }

    /// Ensure every node is running the same build of the software.
    ///
    /// `software_version` is a field of [`CommonConfig`], so the whole-struct
    /// equality check above already catches a mismatch. This runs first purely
    /// for the error message: a version diff is actionable, whereas a diff of
    /// two full config dumps buries the one line that matters.
    pub fn check_software_version(&self) -> Result<()> {
        let mine = self.my_state.common_config.software_version();
        for state in &self.all_states {
            let theirs = state.common_config.software_version();
            ensure!(
                mine == theirs,
                "Software version mismatch across MPC parties: this node is running \
                 {mine}, a peer is running {theirs}. All parties must run the same build \
                 before genesis can proceed."
            );
        }
        Ok(())
    }
}

/// Format a [`SyncState`] for error messages with the potentially huge
/// `excluded_serial_ids` list truncated to a sample (derived `Debug` shows
/// every other field verbatim).
fn summarize_sync_state(state: &SyncState) -> String {
    const SAMPLE: usize = 50;
    let total = state.genesis_config.excluded_serial_ids.len();
    let mut capped = state.clone();
    capped.genesis_config.excluded_serial_ids.truncate(SAMPLE);
    format!("{capped:?} (excluded_serial_ids: {total} total, first {SAMPLE} shown)")
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Config {
        fn new_1() -> Self {
            Self::new(BatchSizeConfig::Static { size: 64 }, vec![3, 5], 100, None)
        }
        fn new_2() -> Self {
            Self::new(
                BatchSizeConfig::Static { size: 64 },
                vec![3, 5, 6],
                200,
                None,
            )
        }
        fn new_3() -> Self {
            Self::new(
                BatchSizeConfig::Static { size: 64 },
                vec![3, 5],
                100,
                Some("abc".to_string()),
            )
        }
    }

    /// `CommonConfig::software_version` is private and populated from a
    /// build-time constant, so a differing version can only be faked through
    /// serde — which is also exactly how a peer's state arrives at runtime.
    fn common_config_with_version(version: &str) -> CommonConfig {
        let mut value = serde_json::to_value(CommonConfig::default()).unwrap();
        value["software_version"] = serde_json::Value::String(version.to_owned());
        serde_json::from_value(value).unwrap()
    }

    impl SyncState {
        fn new_0(genesis_config: Config) -> Self {
            Self::new(CommonConfig::default(), genesis_config)
        }

        fn new_versioned(version: &str) -> Self {
            Self::new(common_config_with_version(version), Config::new_1())
        }

        fn new_1() -> Self {
            Self::new_0(Config::new_1())
        }
        fn new_2() -> Self {
            Self::new_0(Config::new_2())
        }
        fn new_3() -> Self {
            Self::new_0(Config::new_3())
        }
    }

    impl SyncResult {
        fn new_0(states: Vec<SyncState>) -> Self {
            Self::new(states[0].clone(), states)
        }
    }

    #[test]
    fn test_check_genesis_config_all_equal() {
        let result = SyncResult::new_0(vec![
            SyncState::new_1(),
            SyncState::new_1(),
            SyncState::new_1(),
        ]);
        assert!(result.check_synced_state().is_ok());
    }

    #[test]
    fn test_check_genesis_config_not_equal() {
        let result = SyncResult::new_0(vec![
            SyncState::new_1(),
            SyncState::new_2(),
            SyncState::new_2(),
        ]);
        assert!(result.check_synced_state().is_err());
    }

    #[test]
    fn test_check_software_version_all_equal() {
        let result = SyncResult::new_0(vec![
            SyncState::new_versioned("0.1.0+abcdef123456"),
            SyncState::new_versioned("0.1.0+abcdef123456"),
            SyncState::new_versioned("0.1.0+abcdef123456"),
        ]);
        assert!(result.check_synced_state().is_ok());
    }

    #[test]
    fn test_check_software_version_mismatch() {
        let result = SyncResult::new_0(vec![
            SyncState::new_versioned("0.1.0+abcdef123456"),
            SyncState::new_versioned("0.1.0+abcdef123456"),
            SyncState::new_versioned("0.1.0+999999999999"),
        ]);
        let err = result.check_synced_state().unwrap_err().to_string();
        assert!(
            err.contains("Software version mismatch"),
            "genesis must abort with the version-specific error, got: {err}"
        );
    }

    #[test]
    fn test_check_genesis_config_pin_mismatch() {
        let result = SyncResult::new_0(vec![
            SyncState::new_1(),
            SyncState::new_3(),
            SyncState::new_3(),
        ]);
        assert!(result.check_synced_state().is_err());
    }
}
