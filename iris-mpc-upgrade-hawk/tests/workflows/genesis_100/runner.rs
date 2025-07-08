use super::state::TestState;
use crate::utils::{
    runner::{Test, TestContextInfo},
    TestError,
};

pub struct TestRunner {}

impl Test<TestState> for TestRunner {
    async fn exec(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn exec_assert(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn setup(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn setup_assert(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn teardown(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }

    async fn teardown_assert(&self, _ctx: TestContextInfo<TestState>) -> Result<(), TestError> {
        unimplemented!()
    }
}
