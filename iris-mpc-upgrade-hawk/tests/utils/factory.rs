use super::{
    constants, resources,
    types::{NetInputs, NodeInputs, TestRunInfo},
};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns network level inputs for usage during a test run.
pub fn get_net_inputs() -> NetInputs {
    NetInputs::new(
        (0..constants::COUNT_OF_PARTIES)
            .map(|party_id| get_node_inputs(party_id))
            .collect(),
    )
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

    NodeInputs::new(get_args(), resources::read_node_config(party_id).unwrap())
}

/// Returns test run execution information.
pub fn get_test_info(net_inputs: NetInputs) -> TestRunInfo {
    TestRunInfo::new(net_inputs, 100, 0)
}
