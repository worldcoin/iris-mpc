use crate::{
    utils::{resources, TestExecutionEnvironment},
    workflows::genesis_shared::{
        inputs::{SystemStateInputs, TestInputs},
        params::TestParams,
    },
};
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig},
    PartyIdx, PARTY_COUNT, PARTY_IDX_SET,
};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns inputs for running a test.
pub(super) fn create_inputs(params: TestParams) -> TestInputs {
    TestInputs::new(
        create_net_args(params),
        create_net_config(params),
        create_system_state_inputs(params),
    )
}

/// Returns arguments for launching a network.
fn create_net_args(params: TestParams) -> [NodeArgs; PARTY_COUNT] {
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

/// Returns network wide configuration.
fn create_net_config(params: TestParams) -> NetConfig {
    PARTY_IDX_SET
        .iter()
        .map(|party_idx| create_node_config(party_idx, params.node_config_idx()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

/// Returns node specific configuration.
fn create_node_config(party_idx: &PartyIdx, config_idx: usize) -> NodeConfig {
    let fname = format!("node-{}-genesis-{}", party_idx, config_idx);

    resources::read_node_config(fname).unwrap()
}

/// Returns inputs for initializing system state.
fn create_system_state_inputs(params: TestParams) -> SystemStateInputs {
    SystemStateInputs::new(params, vec![])
}
