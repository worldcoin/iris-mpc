use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

// pub enum TestType {
//     // Genesis 100: .
//     Genesis_100,
// }

// Inputs required to run a network.
pub struct NetInputs {
    node_inputs: Vec<NodeInputs>,
}

/// Constructor.
impl NetInputs {
    pub fn new(node_inputs: Vec<NodeInputs>) -> Self {
        Self { node_inputs }
    }
}

// Inputs required to run a node.
pub struct NodeInputs {
    args: NodeArgs,
    config: NodeConfig,
}

/// Constructor.
impl NodeInputs {
    pub fn new(args: NodeArgs, config: NodeConfig) -> Self {
        Self { args, config }
    }
}

/// Encapsulates contextual information pertaining to a test run.
pub struct TestRunInfo {
    inputs: NetInputs,
}

/// Constructor.
impl TestRunInfo {
    pub fn new(inputs: NetInputs) -> Self {
        Self { inputs }
    }
}
