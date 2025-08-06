use super::{factory, types::TestInputs};
use crate::utils::{
    s3_deletions::{get_s3_client, upload_iris_deletions},
    TestError, TestRun, TestRunContextInfo,
};
use eyre::{Report, Result};
use iris_mpc_upgrade_hawk::genesis::exec as exec_genesis;

/// HNSW Genesis test.
pub struct Test {
    /// Data encapsulating test inputs.
    inputs: Option<TestInputs>,

    /// Results of node process execution.
    node_results: Option<Vec<Result<(), Report>>>,
}

/// Constructor.
impl Test {
    pub fn new() -> Self {
        Self {
            inputs: None,
            node_results: None,
        }
    }
}

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
        self.inputs = Some(factory::get_test_inputs(ctx));

        // Write 100 Iris shares -> GPU dB.
        // TODO

        let deleted_serial_ids = vec![];
        let s3_region = "us-east-1";
        let deployment_mode = "dev";
        let s3_client = get_s3_client(Some(s3_region), deployment_mode)
            .await
            .map_err(|e| TestError::SetupError(e.to_string()))?;
        upload_iris_deletions(&deleted_serial_ids, &s3_client, deployment_mode).await?;

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
