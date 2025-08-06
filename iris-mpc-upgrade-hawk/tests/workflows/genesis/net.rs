use super::params::TestParams;
use crate::{
    resources,
    utils::{
        constants::{PARTY_COUNT, PARTY_IDX_SET},
        types::NetConfig,
    },
};
use eyre::{Report, Result};
use iris_mpc_upgrade_hawk::genesis::{
    ExecutionArgs as NodeArgs, ExecutionResult as NodeExecutionResult,
};

// Network wide argument set.
pub type NetArgs = [NodeArgs; PARTY_COUNT];

/// Helper type over set of node execution results.
/// TODO: convert to array -> [Result<NodeExecutionResult, Report>; 3].
pub type NetExecutionResult = Vec<Result<NodeExecutionResult, Report>>;

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
