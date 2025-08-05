use super::{inputs::TestInputs, net::NetArgs, params::TestParams};
use crate::{
    resources,
    utils::{constants::PARTY_IDX_SET, types::NetConfig},
};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Convertor: TestParams -> NetArgs.
impl From<&TestParams> for NetArgs {
    fn from(params: &TestParams) -> Self {
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
impl From<&TestParams> for NetConfig {
    fn from(params: &TestParams) -> Self {
        resources::read_net_config(params.node_config_kind(), params.node_config_idx()).unwrap()
    }
}

/// Convertor: TestParams -> TestInputs.
impl From<&TestParams> for TestInputs {
    fn from(params: &TestParams) -> Self {
        TestInputs::new(
            params.to_owned(),
            NetArgs::from(params),
            NetConfig::from(params),
            resources::read_iris_deletions(params.max_deletions(), 0).unwrap(),
            resources::read_iris_modifications(params.max_modifications(), 0).unwrap(),
        )
    }
}
