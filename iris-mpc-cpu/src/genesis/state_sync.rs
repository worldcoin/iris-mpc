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
        let my_state = self.my_state.clone();
        for state in &self.all_states {
            ensure!(
                *state == my_state,
                "Inconsistent genesis config: \nhave: {:?} \ngot: {:?}",
                my_state,
                state
            );
        }
        Ok(())
    }
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

    impl SyncState {
        fn new_0(genesis_config: Config) -> Self {
            Self::new(CommonConfig::default(), genesis_config)
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
    fn test_check_genesis_config_pin_mismatch() {
        let result = SyncResult::new_0(vec![
            SyncState::new_1(),
            SyncState::new_3(),
            SyncState::new_3(),
        ]);
        assert!(result.check_synced_state().is_err());
    }
}
