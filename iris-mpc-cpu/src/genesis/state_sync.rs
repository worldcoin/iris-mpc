use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, IrisSerialId};
use serde::{Deserialize, Serialize};

/// Encpasulates common Genesis specific configuration information.  This is a network level type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub max_indexation_id: IrisSerialId,
    pub last_indexed_id: IrisSerialId,
    pub excluded_serial_ids: Vec<IrisSerialId>,
}

/// Constructor.
impl Config {
    pub fn new(
        max_indexation_id: IrisSerialId,
        last_indexed_id: IrisSerialId,
        excluded_serial_ids: Vec<IrisSerialId>,
    ) -> Self {
        Self {
            max_indexation_id,
            last_indexed_id,
            excluded_serial_ids,
        }
    }
}

/// Encapsulates a node's synchronization state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncState {
    pub db_len: u64,
    pub common_config: CommonConfig,
    pub genesis_config: Config,
}

/// Constructor.
impl SyncState {
    pub fn new(db_len: u64, common_config: CommonConfig, genesis_config: Config) -> Self {
        Self {
            db_len,
            common_config,
            genesis_config,
        }
    }
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
    pub fn check_genesis_config(&self) -> Result<()> {
        let genesis_config = self.my_state.genesis_config.clone();
        for state in &self.all_states {
            ensure!(
                state.genesis_config == genesis_config,
                "Inconsistent genesis config"
            );
        }
        Ok(())
    }

    /// Check if the genesis-specific state is the same across all nodes.
    pub fn check_common_config(&self) -> Result<()> {
        let my_config = &self.my_state.common_config;
        for SyncState {
            common_config: other_config,
            ..
        } in self.all_states.iter()
        {
            ensure!(
                my_config == other_config,
                "Inconsistent common config!\nhave: {:?}\ngot: {:?}",
                my_config,
                other_config
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
            Self::new(100, 50, vec![3, 5])
        }

        fn new_2() -> Self {
            Self::new(200, 150, vec![3, 5, 6])
        }
    }

    impl SyncState {
        fn new_0(db_len: u64, genesis_config: Config) -> Self {
            Self::new(db_len, CommonConfig::default(), genesis_config)
        }

        fn new_1(db_len: u64) -> Self {
            Self::new_0(db_len, Config::new_1())
        }

        fn new_2(db_len: u64) -> Self {
            Self::new_0(db_len, Config::new_2())
        }
    }

    #[test]
    fn test_check_genesis_config_all_equal() {
        let states = vec![
            SyncState::new_1(10),
            SyncState::new_1(20),
            SyncState::new_1(30),
        ];
        let result = SyncResult::new(states[0].clone(), states);
        assert!(result.check_genesis_config().is_ok());
    }

    #[test]
    fn test_check_genesis_config_not_equal() {
        let states = vec![
            SyncState::new_1(10),
            SyncState::new_2(20),
            SyncState::new_2(30),
        ];
        let result = SyncResult::new(states[0].clone(), states);
        assert!(result.check_genesis_config().is_err());
    }
}
