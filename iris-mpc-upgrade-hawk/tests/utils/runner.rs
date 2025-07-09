use super::{errors::TestError, logger};
use std::fmt;
use std::fmt::Debug;

/// A trait encpasulating a test's workflow lifecycle.
#[allow(async_fn_in_trait)]
pub trait TestRun {
    /// Executes test workflow.
    async fn run(&mut self, ctx: TestRunInfo) -> Result<(), TestError> {
        ctx.log_info("Phase 1.1: Setup");
        self.setup().await?;

        ctx.log_info("Phase 1.2: Setup Assertion");
        self.setup_assert().await?;

        ctx.log_info("Phase 2.1: Execution");
        self.exec().await?;

        ctx.log_info("Phase 2.2: Execution Assertion");
        self.exec_assert().await?;

        ctx.log_info("Phase 3.1: Teardown");
        self.teardown().await?;

        ctx.log_info("Phase 3.2: Teardown Assertion");
        self.teardown_assert().await?;

        Ok(())
    }

    /// Execution phase of a test's workflow.
    async fn exec(&mut self) -> Result<(), TestError>;

    /// Asserts that a test workflow's execution phase was successful.
    async fn exec_assert(&mut self) -> Result<(), TestError>;

    /// Setup phase of a test's workflow.
    async fn setup(&mut self) -> Result<(), TestError>;

    /// Asserts that a test workflow's setup phase was successful.
    async fn setup_assert(&mut self) -> Result<(), TestError>;

    /// Teardown phase of a test's workflow.
    async fn teardown(&mut self) -> Result<(), TestError>;

    /// Asserts that a test workflow's teardown phase was successful.
    async fn teardown_assert(&mut self) -> Result<(), TestError>;
}

/// Metadata associated with a test run.
#[derive(Debug, Clone)]
pub struct TestRunInfo {
    /// Test run ordinal identifier.
    pub idx: usize,

    /// Test kind, e.g. 100.
    pub kind: usize,
}

/// Constructor.
impl TestRunInfo {
    pub fn new(kind: usize) -> Self {
        Self { idx: 1, kind }
    }

    #[allow(dead_code)]
    pub fn new_in_batch(kind: usize, idx: usize) -> Self {
        Self { idx, kind }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for TestRunInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{:02}", self.kind, self.idx)
    }
}

/// Methods.
impl TestRunInfo {
    /// Logs an informational message.
    pub fn log_info(&self, msg: &str) {
        logger::log_info(format!("{}", self).as_str(), msg);
    }
}
