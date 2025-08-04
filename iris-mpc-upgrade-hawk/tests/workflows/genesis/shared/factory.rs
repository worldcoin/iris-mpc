use super::{
    inputs::{SystemStateInputs, TestInputs},
    net::NetArgs,
    params::TestParams,
};
use crate::resources;
use iris_mpc_common::{config::NetConfig, PARTY_IDX_SET};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Convertor: TestParams -> NetArgs.
impl From<TestParams> for NetArgs {
    fn from(params: TestParams) -> Self {
        // TODO: move to static resources (JSON | TOML).
        let args = NodeArgs::new(
            params.batch_size(),
            params.batch_size_error_rate(),
            params.max_indexation_id(),
            params.perform_db_snapshot(),
            params.use_db_backup_as_source(),
        );

        PARTY_IDX_SET
            .iter()
            .map(|_| args.clone())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

/// Convertor: TestParams -> NetConfig.
impl From<TestParams> for NetConfig {
    fn from(params: TestParams) -> Self {
        resources::read_net_config(params.node_config_kind(), params.node_config_idx()).unwrap()
    }
}

/// Convertor: TestParams -> TestInputs.
impl From<TestParams> for TestInputs {
    fn from(params: TestParams) -> Self {
        TestInputs::new(
            NetArgs::from(params),
            NetConfig::from(params),
            SystemStateInputs::from(params),
        )
    }
}

/// Convertor: TestParams -> SystemStateInputs.
impl From<TestParams> for SystemStateInputs {
    fn from(params: TestParams) -> Self {
        let deletions = match params.max_deletions() {
            Some(n_take) => {
                let skip_offset = 0;
                resources::read_iris_deletions(n_take, skip_offset).unwrap()
            }
            None => vec![],
        };

        // TODO: elaborate upon this ... probably the resource loader will
        // return either a stream or a Vec of actual modifications.
        let modifications = match params.max_modifications() {
            Some(max) => resources::read_iris_modifications(max, 0).unwrap(),
            None => vec![],
        };

        SystemStateInputs::new(params, deletions, modifications)
    }
}
