use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;
use std::fmt;

// Inputs required to run a network.
#[derive(Debug, Clone)]
pub struct NetInputs {
    /// Set of node level inputs.
    pub node_inputs: Vec<NodeInputs>,
}

/// Constructor.
impl NetInputs {
    pub fn new(node_inputs: Vec<NodeInputs>) -> Self {
        Self { node_inputs }
    }
}

// Inputs required to run a node.
#[derive(Debug, Clone)]
pub struct NodeInputs {
    /// Node input arguments.
    pub args: NodeArgs,

    /// Node input configuration.
    pub config: NodeConfig,
}

/// Constructor.
impl NodeInputs {
    pub fn new(args: NodeArgs, config: NodeConfig) -> Self {
        Self { args, config }
    }
}

/// Encapsulates contextual information pertaining to a test run.
#[derive(Debug, Clone)]
pub struct TestRunInfo {
    /// Set of network level inputs.
    pub net_inputs: NetInputs,

    /// Type of run, e.g. 100.
    pub run_type: usize,

    /// Test run ordinal identifier.
    pub run_idx: usize,
}

/// Constructor.
impl TestRunInfo {
    pub fn new(inputs: NetInputs, run_type: usize, run_idx: usize) -> Self {
        Self {
            net_inputs: inputs,
            run_type,
            run_idx,
        }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for TestRunInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({:02})", self.run_type, self.run_idx,)
    }
}

/// Methods.
impl TestRunInfo {
    pub fn node_inputs(&self) -> Vec<NodeInputs> {
        self.net_inputs.node_inputs.clone()
    }
}
