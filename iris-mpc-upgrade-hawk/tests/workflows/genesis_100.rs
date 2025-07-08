use crate::utils::{constants::COUNT_OF_PARTIES, runner::TestWorkflow, TestError};
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// HNSW-Genesis-100
///   against:
///     a known set of 100 Iris shares in plaintext format;
///     an empty set of exclusions;
///     an empty set of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
#[derive(Debug, Clone)]
pub struct Workflow {
    state: WorkflowState,
}

pub struct NodeProcessInputs {
    // Arguments to be passed into node process.
    args: [NodeArgs; COUNT_OF_PARTIES],

    // Configurations to be passed into node process.
    configs: [NodeConfig; COUNT_OF_PARTIES],
}

pub struct SystemStateInputs {
    // Serial idenfitiers of deleted Iris's.
    iris_deletions: Vec<IrisSerialId>,

    // Set of Iris shares to be processed.
    iris_shares: Vec<IrisSerialId>,
}

#[derive(Debug, Clone)]
pub struct WorkflowInputs {
    // Arguments to be passed into node process.
    process_inputs: Option<NodeProcessInputs>,

    // Arguments to be passed into node process.
    state_inputs: Option<SystemStateInputs>,
}

impl Default for WorkflowInputs {
    fn default() -> Self {
        Self {
            node_args: None,
            node_configs: None,
            iris_deletions: Vec::new(),
            iris_shares: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkflowState {
    pub inputs: WorkflowInputs,
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self {
            inputs: WorkflowInputs::default(),
        }
    }
}

impl TestWorkflow for Workflow {
    /// Execution phase of a test's workflow.
    async fn exec(&self) -> Result<(), TestError> {
        unimplemented!()
    }

    /// Asserts that a test workflow's execution phase was successful.
    async fn exec_assert(&self) -> Result<(), TestError> {
        unimplemented!()
    }

    /// Setup phase of a test's workflow.
    async fn setup(&self) -> Result<(), TestError> {
        unimplemented!()
    }

    /// Asserts that a test workflow's setup phase was successful.
    async fn setup_assert(&self) -> Result<(), TestError> {
        unimplemented!()
    }

    /// Teardown phase of a test's workflow.
    async fn teardown(&self) -> Result<(), TestError> {
        unimplemented!()
    }

    /// Asserts that a test workflow's teardown phase was successful.
    async fn teardown_assert(&self) -> Result<(), TestError> {
        unimplemented!()
    }
}
