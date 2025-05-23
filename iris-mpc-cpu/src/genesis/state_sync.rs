use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, IrisSerialId};
use serde::{Deserialize, Serialize};

/// Encpasulates common Genesis specific configuration information.  This is a network level type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub max_indexation_id: IrisSerialId,
    pub last_indexed_id: IrisSerialId,
    pub excluded_serial_ids: Vec<IrisSerialId>,
    pub batch_size: usize,
}

/// Encapsulates a node's synchronization state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub common_config: CommonConfig,
    pub genesis_config: Config,
}

/// Encapsulates a result over a node's synchronization state evaluation.
pub struct SyncResult {
    pub my_state: SyncState,
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
                "Inconsistent genesis config!\nhave: {:?}\ngot: {:?}",
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
        pub(crate) fn new_1() -> Self {
            Self {
                max_indexation_id: 100,
                last_indexed_id: 50,
                excluded_serial_ids: vec![3, 5],
                batch_size: 10,
            }
        }

        pub(crate) fn new_2() -> Self {
            Self {
                max_indexation_id: 200,
                last_indexed_id: 150,
                excluded_serial_ids: vec![3, 5, 6],
                batch_size: 20,
            }
        }
    }

    #[test]
    fn test_check_genesis_config_all_equal() {
        let states = vec![
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_1(),
            },
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_1(),
            },
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_1(),
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        assert!(sync_result.check_synced_state().is_ok());
    }

    #[test]
    fn test_check_genesis_config_not_equal() {
        let states = vec![
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_1(),
            },
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_2(),
            },
            SyncState {
                common_config: CommonConfig::default(),
                genesis_config: Config::new_2(),
            },
        ];

        let sync_result = SyncResult::new(states[0].clone(), states);
        assert!(sync_result.check_synced_state().is_err());
    }
}
