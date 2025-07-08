use super::{errors::TestError, logger};
use std::fmt;
use std::fmt::Debug;

/// Encapsulates contextual data pertaining to a test run.
#[derive(Debug, Clone)]
pub struct TestContextInfo<S>
where
    S: Debug + Clone,
{
    /// Test run ordinal identifier.
    pub idx: usize,

    /// Test kind, e.g. 100.
    pub kind: usize,

    /// Test stateful data.
    pub state: Option<S>,
}

/// Constructor.
impl<S> TestContextInfo<S>
where
    S: Debug + Clone,
{
    pub fn new(kind: usize, idx: usize) -> Self {
        Self {
            idx,
            kind,
            state: None,
        }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for TestContextInfo<S>
where
    S: Debug + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({:02})", self.kind, self.idx,)
    }
}

/// Methods.
impl<S> TestContextInfo<S>
where
    S: Debug + Clone,
{
    /// Logs an informational message.
    pub fn log_info(&self, msg: &str) {
        logger::log_info(format!("{}", self).as_str(), msg);
    }
}

/// A trait encpasulating a test's workflow lifecycle.
#[async_trait]
pub trait Test<S>
where
    S: Debug + Clone,
{
    /// Executes test workflow.
    async fn do_test(&self, ctx: TestContextInfo<S>) -> Result<(), TestError> {
        ctx.log_info("Phase 1.1: Setup");
        self.setup(ctx).await?;

        ctx.log_info("Phase 1.2: Setup Assertion");
        self.setup_assert(ctx).await?;

        ctx.log_info("Phase 2.1: Execution");
        self.exec(ctx).await?;

        ctx.log_info("Phase 2.2: Execution Assertion");
        self.exec_assert(ctx).await?;

        ctx.log_info("Phase 3.1: Teardown");
        self.teardown(ctx).await?;

        ctx.log_info("Phase 3.2: Teardown Assertion");
        self.teardown_assert(ctx).await?;

        Ok(())
    }

    /// Execution phase of a test's workflow.
    async fn exec(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;

    /// Asserts that a test workflow's execution phase was successful.
    async fn exec_assert(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;

    /// Setup phase of a test's workflow.
    async fn setup(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;

    /// Asserts that a test workflow's setup phase was successful.
    async fn setup_assert(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;

    /// Teardown phase of a test's workflow.
    async fn teardown(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;

    /// Asserts that a test workflow's teardown phase was successful.
    async fn teardown_assert(&self, ctx: TestContextInfo<S>) -> Result<(), TestError>;
}
