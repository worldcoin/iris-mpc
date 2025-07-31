use super::{
    inputs::{Inputs, NetInputs, NodeInputs, SystemStateInputs},
    params::Params,
};
use crate::utils::{
    constants, resources, DbConnectionInfo, NetConfig, NetDbProvider, NodeDbProvider, NodeDbStore,
    TestExecutionEnvironment, TestRunContextInfo,
};
use eyre::eyre;
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns inputs for running a test.
pub(super) fn create_inputs(ctx: &TestRunContextInfo, params: Params) -> Inputs {
    Inputs::new(
        create_config_net(ctx),
        create_inputs_net(ctx, params),
        create_inputs_of_system_state(params),
    )
}

/// Returns network wide configuration.
fn create_config_net(ctx: &TestRunContextInfo) -> NetConfig {
    [
        create_config_node(ctx.exec_env(), constants::PARTY_IDX_0),
        create_config_node(ctx.exec_env(), constants::PARTY_IDX_1),
        create_config_node(ctx.exec_env(), constants::PARTY_IDX_2),
    ]
}

/// Returns node specific configuration.
fn create_config_node(env: &TestExecutionEnvironment, party_idx: usize) -> NodeConfig {
    let config_fname = format!("node-{}-genesis-0", party_idx);

    resources::read_node_config(env, config_fname).unwrap()
}

/// Returns inputs for launching a network.
fn create_inputs_net(ctx: &TestRunContextInfo, params: Params) -> NetInputs {
    NetInputs::new([
        create_inputs_node(ctx, params, constants::PARTY_IDX_0),
        create_inputs_node(ctx, params, constants::PARTY_IDX_1),
        create_inputs_node(ctx, params, constants::PARTY_IDX_2),
    ])
}

/// Returns inputs for launching a node.
fn create_inputs_node(ctx: &TestRunContextInfo, params: Params, party_idx: usize) -> NodeInputs {
    let args = NodeArgs::new(
        params.batch_size(),
        params.batch_size_error_rate(),
        params.max_indexation_id(),
        params.perform_db_snapshot(),
        params.use_db_backup_as_source(),
    );

    let config_fname = format!("node-{}-genesis-0", party_idx);
    let config = resources::read_node_config(ctx.exec_env(), config_fname).unwrap();

    NodeInputs::new(args, config)
}

/// Returns inputs for initializing system state.
fn create_inputs_of_system_state(params: Params) -> SystemStateInputs {
    SystemStateInputs::new(params, vec![])
}
