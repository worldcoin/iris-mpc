use crate::utils::{
    constants::COUNT_OF_PARTIES,
    resources::{self},
    s3_deletions::{get_s3_client, upload_iris_deletions},
    test_state::{MpcNode, MpcNodes},
    HawkConfigs, TestError, TestRun, TestRunContextInfo,
};
use eyre::Result;
use futures::join;
use iris_mpc_cpu::{
    execution::hawk_main::StoreId,
    genesis::{
        get_iris_deletions,
        plaintext::{GenesisArgs, GenesisState},
    },
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;
/// HNSW Genesis test.
pub struct Test {
    configs: HawkConfigs,

    genesis_args: GenesisArgs,

    rng_state: u64,

    genesis_outputs: Vec<GenesisState>,
}

impl Test {
    pub fn new(genesis_args: GenesisArgs, rng_state: u64) -> Self {
        let exec_env = TestRunContextInfo::new(0, 0);

        let configs = (0..COUNT_OF_PARTIES)
            .map(|idx| {
                resources::read_node_config(&exec_env, format!("node-{idx}-genesis-0")).unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self {
            configs,
            genesis_args,
            rng_state,
            genesis_outputs: vec![],
        }
    }

    async fn get_nodes(&self) -> impl Iterator<Item = MpcNode> {
        MpcNodes::new(&self.configs, self.genesis_args, self.rng_state)
            .await
            .into_iter()
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // these need to be on separate tasks
        let mut join_set = JoinSet::new();
        for config in &self.configs {
            let config = config.clone();
            let genesis_args = self.genesis_args;
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        genesis_args.batch_size,
                        genesis_args.batch_size_error_rate,
                        genesis_args.max_indexation_id,
                        false,
                        false,
                    ),
                    config,
                )
                .await
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap()?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            let max_indexation_id = self.genesis_args.max_indexation_id as usize;
            join_set.spawn(async move {
                let num_irises = node.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_modifications = node.gpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = node.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let iris_stream = node.cpu_iris_store.stream_irises().await;
                /*for expected_pair in &expected_iris_shares {
                    let db_iris: DbStoredIris = iris_stream
                        .next()
                        .await
                        .expect("stream ended early")
                        .expect("failed to get share");
                    // TODO: convert from DbStoredIris to GaloisRingSharedIris and compare
                }*/

                // todo: build plaintext genesis from the .ndjson file
                // todo: compare the plaintext geneis graph to the secret shared cpu graph.
                // maybe also check the iris ids and versions
                let (graph_left, graph_right) = join!(
                    async {
                        let mut graph_tx = node.graph_store.tx().await.unwrap();
                        graph_tx
                            .with_graph(StoreId::Left)
                            .load_to_mem(node.graph_store.pool(), 8)
                            .await
                    },
                    async {
                        let mut graph_tx = node.graph_store.tx().await.unwrap();
                        graph_tx
                            .with_graph(StoreId::Right)
                            .load_to_mem(node.graph_store.pool(), 8)
                            .await
                    }
                );
                let graph_left = graph_left.expect("Could not load left graph");
                let graph_right = graph_right.expect("Could not load right graph");

                //assert!(graph_left == expected.dst_db.graphs[0]);
                //assert!(graph_right == expected.dst_db.graphs[1]);

                // todo: assert persisted_state
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();

        for node in self.get_nodes().await {
            join_set.spawn(async move {
                let plaintext_irises =
                    resources::read_plaintext_iris(0, node.genesis_args.max_indexation_id as usize)
                        .unwrap();
                node.setup_from_plaintext_irises(&plaintext_irises)
                    .await
                    .unwrap()
            });
        }

        while let Some(r) = join_set.join_next().await {
            self.genesis_outputs.push(r.unwrap());
        }

        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            join_set.spawn(async move {
                let deleted_serial_ids = vec![];
                let s3_client = get_s3_client(None, &config.environment).await.unwrap();
                upload_iris_deletions(&deleted_serial_ids, &s3_client, &config.environment)
                    .await
                    .unwrap();
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            let max_indexation_id = node.genesis_args.max_indexation_id as usize;
            join_set.spawn(async move {
                let num_irises = node.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, 0);

                let num_modifications = node.gpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = node.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        // Assert localstack.

        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let max_indexation_id = self.genesis_args.max_indexation_id;
            join_set.spawn(async move {
                let s3_client = get_s3_client(None, &config.environment).await.unwrap();
                let deletions = get_iris_deletions(&config, &s3_client, max_indexation_id)
                    .await
                    .unwrap();
                assert_eq!(deletions.len(), 0);
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        Ok(())
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        Ok(())
    }
}
