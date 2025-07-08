use crate::utils::{runner::TestWorkflow, TestError};

/// HNSW-Genesis-101
///   against:
///     a known set of 100 Iris shares in plaintext format;
///     a known set of exclusions;
///     an empty set of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
pub struct Workflow {}

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
