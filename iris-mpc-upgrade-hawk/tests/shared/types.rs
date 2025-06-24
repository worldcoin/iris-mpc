use iris_mpc_common::config::Config;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;

// Inputs required to run a node.
pub struct NodeExecutionInputs {
    args: ExecutionArgs,
    config: Config,
}

/// Constructor.
impl NodeExecutionInputs {
    pub fn new(args: ExecutionArgs, config: Config) -> Self {
        Self { args, config }
    }
}

/// Encapsulates contextual information pertaining to a test run.
pub struct TestRunContextInfo {
    execution_inputs: Vec<NodeExecutionInputs>,
}

/// Constructor.
impl TestRunContextInfo {
    pub fn new(execution_inputs: Vec<NodeExecutionInputs>) -> Self {
        Self { execution_inputs }
    }
}
