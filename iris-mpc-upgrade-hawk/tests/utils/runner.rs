use eyre::Result;
use std::{
    fmt::{self, Debug},
    path::Path,
};

/// A trait encpasulating a test's workflow lifecycle.
#[allow(async_fn_in_trait)]
pub trait TestRun {
    /// Executes test workflow.
    async fn run(&mut self, ctx: TestRunContextInfo) -> Result<()> {
        tracing::info!(
            "{}: Phase 1.1: Setup (EXECUTION ENV = {:?})",
            ctx,
            ctx.env()
        );
        self.setup(&ctx).await?;

        tracing::info!("{}: Phase 1.2: Setup Assertion", ctx);
        self.setup_assert().await?;

        tracing::info!("{}: Phase 2.1: Execution", ctx);
        self.exec().await?;

        tracing::info!("{}: Phase 2.2: Execution Assertion", ctx);
        self.exec_assert().await?;

        tracing::info!("{}: Phase 3.1: Teardown", ctx);
        self.teardown().await?;

        tracing::info!("{}: Phase 3.2: Teardown Assertion", ctx);
        self.teardown_assert().await?;

        Ok(())
    }

    /// Execution phase of a test's workflow.
    async fn exec(&mut self) -> Result<()>;

    /// Asserts that a test workflow's execution phase was successful.
    async fn exec_assert(&mut self) -> Result<()>;

    /// Setup phase of a test's workflow.
    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        Ok(())
    }

    /// Asserts that a test workflow's setup phase was successful.
    async fn setup_assert(&mut self) -> Result<()> {
        Ok(())
    }

    /// Teardown phase of a test's workflow.
    async fn teardown(&mut self) -> Result<()> {
        Ok(())
    }

    /// Asserts that a test workflow's teardown phase was successful.
    async fn teardown_assert(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Metadata associated with a test run.
#[derive(Debug, Clone, Copy)]
pub struct TestRunContextInfo {
    /// Test run execution environment.
    env: TestRunEnvironment,

    /// Test run ordinal identifier.
    idx: usize,

    /// Test kind, e.g. 100.
    kind: usize,
}

/// Constructor.
impl TestRunContextInfo {
    pub fn new(kind: usize, idx: usize) -> Self {
        Self {
            env: TestRunEnvironment::new(),
            idx,
            kind,
        }
    }
}

/// Accessors.
impl TestRunContextInfo {
    pub fn env(&self) -> &TestRunEnvironment {
        &self.env
    }

    pub fn idx(&self) -> usize {
        self.idx
    }

    pub fn kind(&self) -> usize {
        self.kind
    }
}

/// Trait: fmt::Display.
impl fmt::Display for TestRunContextInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{:02}", self.kind(), self.idx())
    }
}

/// Enumeration over set of test run execution environments.
#[derive(Debug, Clone, Copy)]
pub enum TestRunEnvironment {
    Local,
    Docker,
}

/// Constructor.
impl TestRunEnvironment {
    pub fn new() -> Self {
        if Path::new("/.dockerenv").exists() {
            TestRunEnvironment::Docker
        } else {
            TestRunEnvironment::Local
        }
    }
}
