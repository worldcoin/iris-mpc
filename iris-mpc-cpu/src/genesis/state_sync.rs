use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, IrisSerialId};
use serde::{Deserialize, Serialize};

/// Encpasulates common Genesis specific configuration information.  This is a network level type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    // Size of indexation batch.
    pub batch_size: usize,

    // For Dynamic batch size, this is the error rate for the size calculation.
    pub batch_size_error_rate: usize,

    // Set of identifiers of Iris's to be excluded from indexation.
    pub excluded_serial_ids: Vec<IrisSerialId>,

    // Identifier of last Iris serial ID to have been indexed.
    pub last_indexed_id: IrisSerialId,

    // Identifier of the last Iris serial ID to be indexed.
    pub max_indexation_id: IrisSerialId,

    // Identifier of the last modification ID to be indexed.
    pub max_modification_id: i64,
}

/// Constructor.
impl Config {
    pub fn new(
        batch_size: usize,
        batch_size_error_rate: usize,
        excluded_serial_ids: Vec<IrisSerialId>,
        last_indexed_id: IrisSerialId,
        max_indexation_id: IrisSerialId,
        max_modification_id: i64,
    ) -> Self {
        Self {
            batch_size,
            batch_size_error_rate,
            excluded_serial_ids,
            last_indexed_id,
            max_indexation_id,
            max_modification_id,
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
            Self::new(64, 128, vec![3, 5], 50, 100, 100)
        }
        fn new_2() -> Self {
            Self::new(64, 128, vec![3, 5, 6], 150, 200, 200)
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
}
