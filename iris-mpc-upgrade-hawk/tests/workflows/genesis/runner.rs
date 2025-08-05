use super::{inputs::TestInputs, net::NetExecutionResult, params::TestParams};
use crate::{
    system_state,
    utils::{
        constants::PARTY_IDX_SET,
        types::{NodeType, PartyIdx},
        {pgres::NetDbProvider, TestError, TestRun},
    },
};
use eyre::Result;
use iris_mpc_upgrade_hawk::genesis::exec as exec_genesis;

/// HNSW Genesis test.
pub struct TestRunner {
    /// A dB provider for interacting with various stores.
    db_provider: Option<NetDbProvider>,

    /// Test run inputs.
    inputs: Option<TestInputs>,

    /// Test run parameters.
    params: TestParams,

    /// Execution results over set of nodes being tested.
    results: Option<NetExecutionResult>,
}

/// Constructor.
impl TestRunner {
    pub fn new(params: TestParams) -> Self {
        Self {
            db_provider: None,
            inputs: None,
            results: None,
            params,
        }
    }
}

/// Accessors.
impl TestRunner {
    fn db_provider(&self) -> &NetDbProvider {
        self.db_provider.as_ref().unwrap()
    }

    fn inputs(&self) -> &TestInputs {
        self.inputs.as_ref().unwrap()
    }

    fn params(&self) -> &TestParams {
        &self.params
    }
}

/// Trait: TestRun.
impl TestRun for TestRunner {
    async fn exec(&mut self) -> Result<(), TestError> {
        // Spawn node process futures & await.
        self.results = Some(
            futures::future::join_all(
                PARTY_IDX_SET
                    .into_iter()
                    .map(|node_idx: PartyIdx| {
                        exec_genesis(
                            self.inputs().node_args(node_idx).to_owned(),
                            self.inputs().node_config(node_idx).to_owned(),
                        )
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
        self.inputs = Some(TestInputs::from(self.params()));

        // Set dB provider.
        self.db_provider = Some(NetDbProvider::new_from_config(self.inputs().net_config()).await);

        // Insert Iris shares -> GPU dBs.
        system_state::insert_iris_shares(
            self.db_provider(),
            NodeType::GPU,
            &self.inputs().iris_shares_stream(),
            self.params().shares_pgres_tx_batch_size(),
        )
        .await
        .unwrap();

        // Upload Iris deletions -> AWS S3 buckets.
        system_state::upload_iris_deletions(
            self.inputs().net_config(),
            self.inputs().iris_deletions(),
        )
        .await
        .unwrap();

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        // Assert inputs.
        assert!(&self.inputs.is_some());

        // Assert Iris shares inserted into GPU dB.
        for iris_count in system_state::get_iris_counts(self.db_provider(), &NodeType::GPU)
            .await
            .unwrap()
            .iter()
        {
            assert_eq!(*iris_count, self.params().max_indexation_id() as usize);
        }

        // Assert Iris deletions uploaded to localstack.
        // TODO

        // Assert Iris modifications inserted into CPU dBs.
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
