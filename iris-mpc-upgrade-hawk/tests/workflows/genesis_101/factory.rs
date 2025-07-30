use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

use crate::{
    make_node_configs,
    utils::{constants, resources, NetInputs, SystemStateInputs, TestInputs, TestRunContextInfo},
};

pub fn get_stage1_inputs(ctx: &TestRunContextInfo) -> TestInputs {
    get_inputs(ctx, 50)
}

pub fn get_stage2_inputs(ctx: &TestRunContextInfo) -> TestInputs {
    get_inputs(ctx, 100)
}

fn get_inputs(ctx: &TestRunContextInfo, max_indexation_id: u32) -> TestInputs {
    let args = NodeArgs::new(
        constants::DEFAULT_BATCH_SIZE,
        constants::DEFAULT_BATCH_SIZE_ERROR_RATE,
        max_indexation_id,
        constants::DEFAULT_SNAPSHOT_STRATEGY,
        constants::DEFAULT_BACKUP_AS_SOURCE_STRATEGY,
    );

    let configs = make_node_configs!(|party_id| {
        resources::read_node_config(ctx, format!("node-{}-genesis-0", party_id)).unwrap()
    });

    let system_state_inputs = None;

    TestInputs {
        net_inputs: NetInputs::new(args, configs),
        system_state_inputs,
    }
}
