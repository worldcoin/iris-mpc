use super::{
    factory,
    inputs::{NetProcessInputs, SystemStateInputs},
};
use crate::utils::{TestError, TestRun};

/// HNSW Genesis test.
pub struct Test {
    /// Data encapsulating test inputs.
    inputs: Option<TestInputs>,
}

/// Constructor.
impl Test {
    pub fn new() -> Self {
        Self { inputs: None }
    }
}

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub struct TestInputs {
    // Data used to launch each node process during a test run.
    #[allow(dead_code)]
    net_process_inputs: NetProcessInputs,

    // Data used to initialise system state prior to a test run.
    #[allow(dead_code)]
    system_state_inputs: Option<SystemStateInputs>,
}

/// Constructor.
impl TestInputs {
    pub fn new(
        net_process_inputs: NetProcessInputs,
        system_state_inputs: Option<SystemStateInputs>,
    ) -> Self {
        Self {
            net_process_inputs,
            system_state_inputs,
        }
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        println!("Executing test");

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        println!("Executing test assertion");

        Ok(())
    }

    async fn setup(&mut self) -> Result<(), TestError> {
        // Set inputs.
        self.inputs = Some(TestInputs::new(factory::get_net_process_inputs(), None));

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        println!("Executing setup assertion");

        Ok(())
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        println!("Executing teardown");

        Ok(())
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        println!("Executing teardown assertion");

        Ok(())
    }
}
