use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, IrisSerialId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisConfig {
    pub max_indexation_height: IrisSerialId,
    pub last_indexation_height: IrisSerialId,
    pub excluded_serial_ids: Vec<IrisSerialId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisSyncState {
    pub db_len: u64,
    pub common_config: CommonConfig,
    pub genesis_config: GenesisConfig,
}

pub struct GenesisSyncResult {
    pub my_state: GenesisSyncState,
    pub all_states: Vec<GenesisSyncState>,
}

impl GenesisSyncResult {
    pub fn new(my_state: GenesisSyncState, all_states: Vec<GenesisSyncState>) -> Self {
        Self {
            my_state,
            all_states,
        }
    }

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
        for GenesisSyncState {
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

    #[test]
    fn test_check_genesis_config_all_equal() {
        let genesis_config = GenesisConfig {
            max_indexation_height: 100,
            last_indexation_height: 50,
            excluded_serial_ids: vec![3, 5],
        };

        let states = vec![
            GenesisSyncState {
                db_len: 10,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config.clone(),
            },
            GenesisSyncState {
                db_len: 20,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config.clone(),
            },
            GenesisSyncState {
                db_len: 30,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config.clone(),
            },
        ];

        let sync_result = GenesisSyncResult::new(states[0].clone(), states);
        assert!(sync_result.check_genesis_config().is_ok());
    }

    #[test]
    fn test_check_genesis_config_not_equal() {
        let genesis_config_1 = GenesisConfig {
            max_indexation_height: 100,
            last_indexation_height: 50,
            excluded_serial_ids: vec![3, 5],
        };
        let genesis_config_2 = GenesisConfig {
            max_indexation_height: 200,
            last_indexation_height: 150,
            excluded_serial_ids: vec![3, 5, 6],
        };

        let states = vec![
            GenesisSyncState {
                db_len: 10,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config_1.clone(),
            },
            GenesisSyncState {
                db_len: 20,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config_2.clone(),
            },
            GenesisSyncState {
                db_len: 30,
                common_config: CommonConfig::default(),
                genesis_config: genesis_config_1.clone(),
            },
        ];

        let sync_result = GenesisSyncResult::new(states[0].clone(), states);
        assert!(sync_result.check_genesis_config().is_err());
    }
}
