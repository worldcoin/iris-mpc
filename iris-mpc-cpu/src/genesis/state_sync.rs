use super::BatchSizeConfig;
use eyre::{ensure, Result};
use iris_mpc_common::{config::CommonConfig, helpers::sync::Modification, IrisSerialId};
use serde::{Deserialize, Serialize};

/// Encapsulates common Genesis specific configuration information. This is a network level type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    // Batch size configuration (static or dynamic with cap).
    pub batch_size_config: BatchSizeConfig,

    // Set of identifiers of Iris's to be excluded from indexation.
    pub excluded_serial_ids: Vec<IrisSerialId>,

    // Identifier of last Iris serial ID to have been indexed.
    pub last_indexed_id: IrisSerialId,

    // Identifier of the last Iris serial ID to be indexed.
    pub max_indexation_id: IrisSerialId,

    // Identifier of the last modification ID to be indexed.
    pub max_modification_id: i64,

    // Identifier of the max completed modification the node has seen
    pub max_modification_persisted_id: i64,

    // The modifications that will be performed during the delta phase
    pub modifications: Vec<Modification>,
}

/// Constructor.
impl Config {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        batch_size_config: BatchSizeConfig,
        excluded_serial_ids: Vec<IrisSerialId>,
        last_indexed_id: IrisSerialId,
        max_indexation_id: IrisSerialId,
        max_modification_id: i64,
        max_modification_persisted_id: i64,
        modifications: Vec<Modification>,
    ) -> Self {
        Self {
            batch_size_config,
            excluded_serial_ids,
            last_indexed_id,
            max_indexation_id,
            max_modification_id,
            max_modification_persisted_id,
            modifications,
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
    use std::vec;

    use iris_mpc_common::helpers::{
        smpc_request::{IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE},
        sync::Modification,
    };

    use super::*;

    impl Config {
        fn new_1() -> Self {
            let mod_1 = Modification {
                id: 1,
                serial_id: Some(1),
                request_type: REAUTH_MESSAGE_TYPE.to_string(),
                s3_url: Some("s3_url_1".to_string()),
                status: "status_1".to_string(),
                persisted: false,
                result_message_body: None,
                graph_mutation: None,
            };
            let mod_2 = Modification {
                id: 2,
                serial_id: Some(2),
                request_type: IDENTITY_DELETION_MESSAGE_TYPE.to_string(),
                s3_url: Some("s3_url_2".to_string()),
                status: "status_2".to_string(),
                persisted: false,
                result_message_body: None,
                graph_mutation: None,
            };
            Self::new(
                BatchSizeConfig::Static { size: 64 },
                vec![3, 5],
                50,
                100,
                100,
                10,
                vec![mod_1, mod_2],
            )
        }
        fn new_2() -> Self {
            let mod_1 = Modification {
                id: 1,
                serial_id: Some(1),
                request_type: REAUTH_MESSAGE_TYPE.to_string(),
                s3_url: Some("s3_url_1".to_string()),
                status: "status_1".to_string(),
                persisted: false,
                result_message_body: Some("meow".to_string()),
                graph_mutation: None,
            };
            Self::new(
                BatchSizeConfig::Static { size: 64 },
                vec![3, 5, 6],
                150,
                200,
                200,
                200,
                vec![mod_1],
            )
        }

        fn new_3() -> Self {
            let mod_1 = Modification {
                id: 1,
                serial_id: Some(1),
                request_type: REAUTH_MESSAGE_TYPE.to_string(),
                s3_url: Some("s3_url_1".to_string()),
                status: "status_1".to_string(),
                persisted: false,
                result_message_body: Some("hello".to_string()),
                graph_mutation: None,
            };
            Self::new(
                BatchSizeConfig::Static { size: 64 },
                vec![3, 5, 6],
                150,
                200,
                200,
                200,
                vec![mod_1],
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
    fn test_check_genesis_config_modifications_equal_except_body() {
        let result = SyncResult::new_0(vec![
            SyncState::new_3(),
            SyncState::new_2(),
            SyncState::new_2(),
        ]);
        assert!(result.check_synced_state().is_ok());
    }
}
