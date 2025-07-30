use super::{
    inputs::{Inputs, NetInputs, NodeProcessInputs, SystemStateInputs},
    params::Params,
};
use crate::utils::{constants, resources, TestRunContextInfo};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns test run inputs.
pub(super) fn get_test_inputs(ctx: &TestRunContextInfo, params: Params) -> Inputs {
    Inputs::new(get_net_inputs(ctx, params), get_system_state_inputs(params))
}

/// Returns inputs for running network.
fn get_net_inputs(ctx: &TestRunContextInfo, params: Params) -> NetInputs {
    NetInputs::new([
        get_node_process_inputs(ctx, params, constants::PARTY_IDX_0),
        get_node_process_inputs(ctx, params, constants::PARTY_IDX_1),
        get_node_process_inputs(ctx, params, constants::PARTY_IDX_2),
    ])
}

/// Returns inputs for running a node.
fn get_node_process_inputs(
    ctx: &TestRunContextInfo,
    params: Params,
    party_id: usize,
) -> NodeProcessInputs {
    let args = NodeArgs::new(
        params.batch_size(),
        params.batch_size_error_rate(),
        params.max_indexation_id(),
        params.perform_db_snapshot(),
        params.use_db_backup_as_source(),
    );
    let config = resources::read_node_config(ctx, format!("node-{}-genesis-0", party_id)).unwrap();

    NodeProcessInputs::new(args, config)
}

/// Returns inputs for initializing system state.
fn get_system_state_inputs(params: Params) -> SystemStateInputs {
    SystemStateInputs::new(params, vec![])
}
