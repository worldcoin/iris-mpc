use super::{factory, state_mutator};
use crate::{
    utils::{TestError, TestRun, TestRunContextInfo},
    workflows::genesis_shared::{
        inputs::TestInputs,
        net::{NetArgs, NetExecutionResult},
        params::TestParams,
    },
};
use eyre::Result;
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig},
    PartyIdx, PARTY_IDX_SET,
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs as NodeArgs};

/// HNSW Genesis test.
pub struct Test {
    /// Test run inputs.
    inputs: Option<TestInputs>,

    /// Test run parameters.
    params: TestParams,

    /// Node execution results.
    results: Option<NetExecutionResult>,
}

/// Constructor.
impl Test {
    pub fn new(params: TestParams) -> Self {
        Self {
            inputs: None,
            results: None,
            params,
        }
    }
}

/// Accessors.
impl Test {
    pub fn net_args(&self) -> &NetArgs {
        self.inputs.as_ref().unwrap().net_args()
    }

    pub fn net_config(&self) -> &NetConfig {
        self.inputs.as_ref().unwrap().net_config()
    }

    pub fn node_args(&self, node_idx: PartyIdx) -> NodeArgs {
        self.inputs.as_ref().unwrap().node_args(node_idx).clone()
    }

    pub fn node_config(&self, node_idx: PartyIdx) -> NodeConfig {
        self.inputs.as_ref().unwrap().node_config(node_idx).clone()
    }

    pub fn params(&self) -> &TestParams {
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
                exec_genesis(self.node_args(node_idx), self.node_config(node_idx))
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
                    return Err(TestError::NodePanicError(node_idx, err.to_string()));
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
        state_mutator::insert_iris_shares_into_gpu_stores(self.net_config(), self.params()).await;

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
