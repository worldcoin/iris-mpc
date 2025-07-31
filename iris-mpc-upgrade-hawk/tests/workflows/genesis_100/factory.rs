use super::{
    inputs::{Inputs, SystemStateInputs},
    params::Params,
};
use crate::utils::{constants::COUNT_OF_PARTIES, resources, TestExecutionEnvironment};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns inputs for running a test.
pub(super) fn create_inputs(env: &TestExecutionEnvironment, params: Params) -> Inputs {
    Inputs::new(
        create_net_args(params),
        create_net_config(env),
        create_system_state_inputs(params),
    )
}

/// Returns inputs for launching a node.
fn create_net_args(params: Params) -> [NodeArgs; COUNT_OF_PARTIES] {
    let args = NodeArgs::new(
        params.batch_size(),
        params.batch_size_error_rate(),
        params.max_indexation_id(),
        params.perform_db_snapshot(),
        params.use_db_backup_as_source(),
    );

    (0..COUNT_OF_PARTIES)
        .map(|_| args.clone())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Returns network wide configuration.
fn create_net_config(env: &TestExecutionEnvironment) -> [NodeConfig; COUNT_OF_PARTIES] {
    (0..COUNT_OF_PARTIES)
        .map(|party_idx| create_node_config(env, party_idx))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Returns node specific configuration.
fn create_node_config(env: &TestExecutionEnvironment, party_idx: usize) -> NodeConfig {
    let fname = format!("node-{}-genesis-0", party_idx);

    resources::read_node_config(env, fname).unwrap()
}

/// Returns inputs for initializing system state.
fn create_system_state_inputs(params: Params) -> SystemStateInputs {
    SystemStateInputs::new(params, vec![])
}
