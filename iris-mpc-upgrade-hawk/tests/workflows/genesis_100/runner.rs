use super::{
    factory,
    inputs::{Inputs, NetArgs},
    params::Params,
    state_mutator,
};
use crate::utils::{TestError, TestRun, TestRunContextInfo};
use eyre::{Report, Result};
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig},
    PartyIdx, PARTY_IDX_SET,
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs as NodeArgs};

/// HNSW Genesis test.
pub struct Test {
    /// Test run inputs.
    inputs: Option<Inputs>,

    /// Test run parameters.
    params: Params,

    /// Node execution results.
    results: Option<Vec<Result<(), Report>>>,
}

/// Constructor.
impl Test {
    pub fn new(params: Params) -> Self {
        Self {
            inputs: None,
            results: None,
            params,
        }
    }
}

/// Accessors.
impl Test {
    pub fn args(&self) -> &NetArgs {
        self.inputs.as_ref().unwrap().args()
    }

    pub fn args_of_node(&self, node_idx: PartyIdx) -> NodeArgs {
        self.inputs.as_ref().unwrap().args_of_node(node_idx).clone()
    }

    pub fn config(&self) -> &NetConfig {
        self.inputs.as_ref().unwrap().config()
    }

    pub fn config_of_node(&self, node_idx: PartyIdx) -> NodeConfig {
        self.inputs
            .as_ref()
            .unwrap()
            .config_of_node(node_idx)
            .clone()
    }

    pub fn params(&self) -> &Params {
        &self.params
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // Set node process futures.
        let node_futures: Vec<_> = PARTY_IDX_SET
            .into_iter()
            .map(|node_idx: PartyIdx| {
                exec_genesis(self.args_of_node(node_idx), self.config_of_node(node_idx))
            })
            .collect();

        // Await futures to complete.
        self.results = Some(futures::future::join_all(node_futures).await);

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        // Assert node results.
        for (node_idx, result) in self.results.as_ref().unwrap().iter().enumerate() {
            match result {
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
        // ... insert Iris shares -> GPU dB.
        state_mutator::insert_iris_shares_into_gpu_stores(self.config(), self.params()).await;

        // ... insert Iris deletions -> AWS S3.
        // state_mutator::insert_iris_deletions(self.params(), self.args(), self.config()).await;

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
