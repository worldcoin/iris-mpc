use std::sync::Arc;

use super::params::Params;
use crate::utils::{
    constants::COUNT_OF_PARTIES, resources, s3_client, HawkConfigs, NetDbProvider, TestError,
    TestExecutionEnvironment, TestRun, TestRunContextInfo,
};
use eyre::Result;
use futures::join;
use iris_mpc_cpu::{
    execution::hawk_main::StoreId,
    genesis::{
        get_iris_deletions,
        plaintext::{GenesisArgs, GenesisConfig, GenesisState},
    },
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

/// HNSW Genesis test.
pub struct Test {
    configs: HawkConfigs,

    /// Test run parameters.
    params: Params,

    genesis_args: ExecutionArgs,
}

/// Constructor.
impl Test {
    pub fn new(params: Params) -> Self {
        let exec_env = TestExecutionEnvironment::new();

        let configs = (0..COUNT_OF_PARTIES)
            .map(|idx| {
                resources::read_node_config(&exec_env, format!("node-{idx}-genesis-0")).unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let genesis_args = params.to_genesis_execution_args();

        Self {
            configs,
            params,
            genesis_args,
        }
    }
}

/// Trait: TestRun.
impl TestRun for Test {
    async fn exec(&mut self) -> Result<(), TestError> {
        // these need to be on separate tasks
        let mut join_set = JoinSet::new();
        for config in &self.configs {
            let config = config.clone();
            let genesis_args = self.genesis_args.clone();
            join_set.spawn(async move { exec_genesis(genesis_args, config).await });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap()?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        let db_provider = NetDbProvider::new_from_config(&self.configs).await;

        let expected: Arc<GenesisState> = {
            let db = db_provider.iter().next().unwrap();
            let config = &self.configs[0];
            let r = db
                .simulate_genesis(
                    GenesisConfig {
                        hnsw_M: config.hnsw_param_M,
                        hnsw_ef_constr: config.hnsw_param_ef_constr,
                        hnsw_ef_search: config.hnsw_param_ef_search,
                        hawk_prf_key: Some(self.params.rng_state()),
                    },
                    GenesisArgs {
                        max_indexation_id: self.params.max_indexation_id(),
                        batch_size: self.params.batch_size(),
                        batch_size_error_rate: self.params.batch_size_error_rate(),
                    },
                )
                .await
                .unwrap();
            Arc::new(r)
        };

        let mut join_set = JoinSet::new();
        for db in db_provider.into_iter() {
            let expected = expected.clone();
            let max_indexation_id = self.params.max_indexation_id() as usize;
            join_set.spawn(async move {
                let num_irises = db.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = db.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_modifications = db.gpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = db.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let (graph_left, graph_right) = join!(
                    async {
                        let mut graph_tx = db.graph_store.tx().await.unwrap();
                        graph_tx
                            .with_graph(StoreId::Left)
                            .load_to_mem(db.graph_store.pool(), 8)
                            .await
                    },
                    async {
                        let mut graph_tx = db.graph_store.tx().await.unwrap();
                        graph_tx
                            .with_graph(StoreId::Right)
                            .load_to_mem(db.graph_store.pool(), 8)
                            .await
                    }
                );
                let graph_left = graph_left.expect("Could not load left graph");
                let graph_right = graph_right.expect("Could not load right graph");

                assert!(graph_left == expected.dst_db.graphs[0]);
                assert!(graph_right == expected.dst_db.graphs[1]);

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
        for (db, shares) in itertools::izip!(
            NetDbProvider::new_from_config(&self.configs)
                .await
                .into_iter(),
            resources::read_iris_shares(
                self.params.rng_state(),
                0,
                self.params.max_indexation_id() as usize,
            )
            .unwrap()
            .into_iter()
        ) {
            join_set.spawn(async move {
                db.init_iris_stores(shares.as_slice()).await.unwrap();
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            join_set.spawn(async move {
                let aws_clients = iris_mpc::services::aws::clients::AwsClients::new(&config)
                    .await
                    .expect("failed to create aws clients");
                s3_client::clear_s3_iris_deletions(&config, &aws_clients.s3_client)
                    .await
                    .expect("failed to clear iris deletions");
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        let mut join_set = JoinSet::new();
        for db in NetDbProvider::new_from_config(&self.configs)
            .await
            .into_iter()
        {
            let max_indexation_id = self.params.max_indexation_id() as usize;
            join_set.spawn(async move {
                let num_irises = db.gpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = db.cpu_iris_store.count_irises().await.unwrap();
                assert_eq!(num_irises, 0);

                let num_modifications = db.gpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = db.cpu_iris_store.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        // Assert localstack.

        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let max_indexation_id = self.params.max_indexation_id();
            join_set.spawn(async move {
                let aws_clients = iris_mpc::services::aws::clients::AwsClients::new(&config)
                    .await
                    .unwrap();
                let deletions =
                    get_iris_deletions(&config, &aws_clients.s3_client, max_indexation_id)
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
