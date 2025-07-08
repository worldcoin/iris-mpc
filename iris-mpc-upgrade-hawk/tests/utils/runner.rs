use super::{errors::TestError, logger, types::TestRunInfo};
use iris_mpc_upgrade_hawk::genesis::exec as exec_genesis;
use std::fmt;

/// Encapsulates contextual information pertaining to a test run.
#[derive(Debug, Clone)]
pub struct TestContextInfo {
    /// Type of run, e.g. 100.
    pub run_type: usize,

    /// Test run ordinal identifier.
    pub run_idx: usize,
}

impl TestContextInfo {
    pub fn log_info(&self, msg: &str) {
        logger::log_info(format!("{}", self).as_str(), msg.to_string());
    }
}

/// Trait: fmt::Display.
impl fmt::Display for TestContextInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} ({:02})", self.run_type, self.run_idx,)
    }
}

/// A trait encpasulating a test's workflow lifecycle.
pub trait TestWorkflow {
    /// Execution phase of a test's workflow.
    async fn exec(&self) -> Result<(), TestError>;

    /// Asserts that a test workflow's execution phase was successful.
    async fn exec_assert(&self) -> Result<(), TestError>;

    /// Executes test workflow.
    async fn exec_test(&self, ctx: TestContextInfo) -> Result<(), TestError> {
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

    /// Setup phase of a test's workflow.
    async fn setup(&self) -> Result<(), TestError>;

    /// Asserts that a test workflow's setup phase was successful.
    async fn setup_assert(&self) -> Result<(), TestError>;

    /// Teardown phase of a test's workflow.
    async fn teardown(&self) -> Result<(), TestError>;

    /// Asserts that a test workflow's teardown phase was successful.
    async fn teardown_assert(&self) -> Result<(), TestError>;
}

pub async fn exec_test(info: TestRunInfo) {
    logger::log_info(
        format!("{}", info).as_str(),
        "Test execution starts".to_string(),
    );

    // Set collected futures.
    let exec_futures: Vec<_> = info
        .node_inputs()
        .into_iter()
        .map(|node_input| {
            let args = node_input.args;
            let config = node_input.config;
            exec_genesis(args, config)
        })
        .collect();

    // Await all futures to complete.
    let exec_results = futures::future::join_all(exec_futures).await;

    for result in exec_results {
        if let Err(e) = result {
            eprintln!("Error during execution: {}", e);
        }
    }

    logger::log_info(
        format!("{}", info).as_str(),
        "Test execution ends".to_string(),
    );
}
