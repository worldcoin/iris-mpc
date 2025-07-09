use super::inputs::TestInputs;
use crate::utils::{runner::TestRun, TestError};

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

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn setup(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        unimplemented!()
    }
}
