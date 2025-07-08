use crate::utils::{constants::COUNT_OF_PARTIES, types::NodeInputs};
use iris_mpc_common::IrisSerialId;

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub struct TestInputs {
    // Data used to launch each node process during a test run.
    node_process_inputs: NodeProcessInputs,

    // Data used to initialise system state prior to a test run.
    system_state_inputs: SystemStateInputs,
}

/// Encapsulates data used to launch each node process during a test run.
pub type NodeProcessInputs = [NodeInputs; COUNT_OF_PARTIES];

/// Encapsulates data used to initialise system state prior to a test run.
#[derive(Debug, Clone)]
pub struct SystemStateInputs {
    // Serial idenfitiers of deleted Iris's.
    iris_deletions: Vec<IrisSerialId>,

    // Set of Iris shares to be processed.
    iris_shares: Vec<IrisSerialId>,
}
