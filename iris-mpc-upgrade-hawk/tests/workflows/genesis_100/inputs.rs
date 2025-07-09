use crate::utils::constants::COUNT_OF_PARTIES;
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Inputs required to run a network.
#[derive(Debug, Clone)]
pub struct NetProcessInputs {
    /// Node input arguments.
    pub node_process_inputs: [NodeProcessInputs; COUNT_OF_PARTIES],
}

/// Constructor.
impl NetProcessInputs {
    pub fn new(node_process_inputs: [NodeProcessInputs; COUNT_OF_PARTIES]) -> Self {
        Self {
            node_process_inputs,
        }
    }
}

/// Inputs required to run a node.
#[derive(Debug, Clone)]
pub struct NodeProcessInputs {
    /// Node input arguments.
    pub args: NodeArgs,

    /// Node input configuration.
    pub config: NodeConfig,
}

/// Constructor.
impl NodeProcessInputs {
    pub fn new(args: NodeArgs, config: NodeConfig) -> Self {
        Self { args, config }
    }
}

/// Encapsulates data used to initialise system state prior to a test run.
#[derive(Debug, Clone)]
pub struct SystemStateInputs {
    // Serial idenfitiers of deleted Iris's.
    iris_deletions: Vec<IrisSerialId>,

    // Set of Iris shares to be processed.
    iris_shares: Vec<IrisSerialId>,
}
