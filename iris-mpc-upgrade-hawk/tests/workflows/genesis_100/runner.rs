use super::{factory, inputs::Inputs, params::Params};
use crate::utils::{constants::COUNT_OF_PARTIES, TestError, TestRun, TestRunContextInfo};
use eyre::{Report, Result};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs as NodeArgs};

/// HNSW Genesis test.
pub struct Test {
    /// Data encapsulating test inputs.
    inputs: Option<Inputs>,

    /// Results of node process execution.
    node_results: Option<Vec<Result<(), Report>>>,

    /// Results of node process execution.
    params: Params,
}

/// Constructor.
impl Test {
    pub fn new(params: Params) -> Self {
        Self {
            inputs: None,
            node_results: None,
            params,
        }
    }
}

/// Accessors.
impl Test {
    pub fn node_args(&self, node_idx: usize) -> &NodeArgs {
        self.inputs.as_ref().unwrap().args_of_node(node_idx)
    }

    pub fn node_config(&self, node_idx: usize) -> &NodeConfig {
        self.inputs.as_ref().unwrap().config_of_node(node_idx)
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // Set node process futures.
        let node_futures: Vec<_> = (0..COUNT_OF_PARTIES)
            .map(|node_idx: usize| {
                let node_config = self.node_config(node_idx);
                let node_args = self.node_args(node_idx);
                exec_genesis(node_args.to_owned(), node_config.to_owned())
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
        self.inputs = Some(factory::create_inputs(ctx.exec_env(), self.params));

        // Set system state.
        // Write Iris shares -> GPU dB.
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
