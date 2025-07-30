use crate::{
    utils::{TestError, TestInputs, TestRun, TestRunContextInfo},
    workflows::genesis_101::factory,
};
use derive_more::{Deref, DerefMut};
use eyre::{eyre, Report, Result};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade_hawk::genesis::exec as exec_genesis;
use std::ops::{Deref, DerefMut};

#[derive(Deref, DerefMut, Default)]
pub struct Test(crate::utils::Test);

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // Set node process inputs.
        let node_inputs = self
            .inputs
            .as_ref()
            .unwrap()
            .net_inputs()
            .node_process_inputs()
            .iter();

        // Set node process futures.
        let node_futures: Vec<_> = node_inputs
            .map(|node_input| {
                exec_genesis(node_input.args().to_owned(), node_input.config().to_owned())
            })
            .collect();

        // Await futures to complete.
        self.node_results = Some(futures::future::join_all(node_futures).await);

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        // Assert node process results.
        for (node_idx, node_result) in self.node_results.as_ref().unwrap().iter().enumerate() {
            match node_result {
                Ok(_) => (),
                Err(err) => {
                    return Err(TestError::NodeProcessPanicError(node_idx, err.to_string()));
                }
            }
        }

        // Assert CPU dB tables: iris, hawk_graph_entry, hawk_graph_links, persistent_state
        // TODO

        Ok(())
    }

    async fn setup(&mut self, ctx: &TestRunContextInfo) -> Result<(), TestError> {
        // Set inputs.
        self.inputs = Some(factory::get_stage1_inputs(ctx));

        // todo: self.get_db_contexts?.map(|x| x.init_modifications_from_file(...));

        // Write 100 Iris shares -> GPU dB.
        // TODO

        // Write empty Iris deletions -> localstack.
        // TODO

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        // Assert inputs.
        assert!(&self.inputs.is_some());

        // Assert dBs.
        // TODO

        // Assert localstack.
        // TODO

        Ok(())
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        Ok(())
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        Ok(())
    }
}
