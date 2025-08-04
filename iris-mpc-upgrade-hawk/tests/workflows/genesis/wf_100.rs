use super::shared::{inputs::TestInputs, net::NetExecutionResult, params::TestParams};
use crate::{
    resources, system_state,
    utils::{pgres::NetDbProvider, TestError, TestRun},
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
    pub fn inputs(&self) -> &TestInputs {
        self.inputs.as_ref().unwrap()
    }

    pub fn net_config(&self) -> &NetConfig {
        self.inputs().net_config()
    }

    pub fn node_args(&self, node_idx: PartyIdx) -> NodeArgs {
        self.inputs().node_args(node_idx).clone()
    }

    pub fn node_config(&self, node_idx: PartyIdx) -> NodeConfig {
        self.inputs().node_config(node_idx).clone()
    }

    pub fn params(&self) -> &TestParams {
        &self.params
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // Spawn node process futures & await.
        self.results = Some(
            futures::future::join_all(
                PARTY_IDX_SET
                    .into_iter()
                    .map(|node_idx: PartyIdx| {
                        exec_genesis(self.node_args(node_idx), self.node_config(node_idx))
                    })
                    .collect::<Vec<_>>(),
            )
            .await,
        );

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

    async fn setup(&mut self) -> Result<(), TestError> {
        // Set inputs.
        self.inputs = Some(TestInputs::from(self.params));

        // Insert Iris shares -> GPU dB.
        insert_iris_shares_into_gpu_store(self.net_config(), self.params())
            .await
            .unwrap();

        // Upload Iris deletions -> AWS S3.
        upload_iris_deletions(self.net_config(), self.inputs())
            .await
            .unwrap();

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

/// Inserts Iris shares into GPU store.
async fn insert_iris_shares_into_gpu_store(
    net_config: &NetConfig,
    params: &TestParams,
) -> Result<()> {
    // Set shares batch generator.
    let batch_size = params.shares_generator_batch_size();
    let read_maximum = params.max_indexation_id() as usize;
    let rng_state = params.shares_generator_rng_state();
    let skip_offset = 0;
    let batch_generator =
        resources::read_iris_shares_batch(batch_size, read_maximum, rng_state, skip_offset)
            .unwrap();

    // Iterate over batches and insert into GPU store.
    // TODO: process serial id ranges.
    let db_provider = NetDbProvider::new_from_config(net_config).await;
    let tx_batch_size = params.shares_pgres_tx_batch_size();
    let _ = system_state::insert_iris_shares(&batch_generator, &db_provider, tx_batch_size)
        .await
        .unwrap();

    // TODO: process serial id ranges.

    Ok(())
}

/// Uploads Iris deletions into AWS S3 bucket.
async fn upload_iris_deletions(net_config: &NetConfig, inputs: &TestInputs) -> Result<()> {
    system_state::upload_iris_deletions(net_config, inputs.system_state_inputs().iris_deletions())
        .await
        .unwrap();

    Ok(())
}
