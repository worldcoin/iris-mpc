use super::{
    constants,
    types::{NetInputs, NodeInputs},
};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns network level inputs for usage during a test run.
pub fn get_net_inputs() -> NetInputs {
    NetInputs::new(vec![
        get_node_inputs(0),
        get_node_inputs(1),
        get_node_inputs(2),
    ])
}

/// Returns node level inputs for usage during a test run.
pub fn get_node_inputs(party_id: usize) -> NodeInputs {
    fn get_args() -> NodeArgs {
        NodeArgs::new(
            constants::DEFAULT_MAX_INDEXATION_ID,
            constants::DEFAULT_BATCH_SIZE,
            constants::DEFAULT_BATCH_SIZE_ERROR_RATE,
            constants::DEFAULT_SNAPSHOT_STRATEGY,
        )
    }

    pub fn get_config(_party_id: usize) -> NodeConfig {
        unimplemented!()
    }

    NodeInputs::new(get_args(), get_config(party_id))
}
