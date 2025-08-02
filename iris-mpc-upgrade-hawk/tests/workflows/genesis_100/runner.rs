use super::params::Params;
use crate::utils::{
    constants::COUNT_OF_PARTIES, resources, s3_client, HawkConfigs, NetDbProvider, TestError,
    TestExecutionEnvironment, TestRun, TestRunContextInfo,
};
use eyre::Result;
use futures::{future::join_all, join};
use iris_mpc_cpu::{
    execution::hawk_main::StoreId,
    genesis::{
        get_iris_deletions,
        plaintext::{GenesisArgs, GenesisConfig},
    },
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

macro_rules! join_all_and_report_errors {
    ($futures:expr, $err_msg:expr) => {{
        let results = join_all($futures).await;
        for (idx, result) in results.iter().enumerate() {
            if let Err(e) = result {
                panic!("{}: Error from Node {}: {:?}", $err_msg, idx, e);
            }
        }
        results
    }};
}

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
        for (db, config) in itertools::izip!(db_provider.iter(), self.configs.iter()) {
            let num_irises = db.gpu_iris_store.count_irises().await?;
            assert_eq!(num_irises, self.params.max_indexation_id() as usize);

            let num_irises = db.cpu_iris_store.count_irises().await?;
            assert_eq!(num_irises, self.params.max_indexation_id() as usize);

            let num_modifications = db.gpu_iris_store.last_modifications(1).await?;
            assert_eq!(num_modifications.len(), 0);

            let num_modifications = db.cpu_iris_store.last_modifications(1).await?;
            assert_eq!(num_modifications.len(), 0);

            // build a graph from non secret shared irises using the same parameters and check
            // if genesis produced the same output
            let expected = db
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
                .await?;

            let (graph_left, graph_right) = join!(
                async {
                    let mut graph_tx = db.graph_store.tx().await?;
                    graph_tx
                        .with_graph(StoreId::Left)
                        .load_to_mem(db.graph_store.pool(), 8)
                        .await
                },
                async {
                    let mut graph_tx = db.graph_store.tx().await?;
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
        }

        // todo: assert persisted_state

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<(), TestError> {
        let shares = resources::read_iris_shares(
            self.params.rng_state(),
            0,
            self.params.max_indexation_id() as usize,
        )
        .unwrap();

        let db_provider = NetDbProvider::new_from_config(&self.configs).await;

        // initialize iris and graph stores
        let futs: Vec<_> = itertools::izip!(db_provider.iter(), &shares)
            .map(|(db, shares)| db.init_iris_stores(shares.as_slice()))
            .collect();
        join_all_and_report_errors!(futs, "failed to init stores");

        // clear Iris deletions -> AWS S3.
        for config in self.configs.iter() {
            let aws_clients = iris_mpc::services::aws::clients::AwsClients::new(config)
                .await
                .expect("failed to create aws clients");
            s3_client::clear_s3_iris_deletions(config, &aws_clients.s3_client)
                .await
                .expect("failed to clear iris deletions");
        }

        Ok(())
    }

    async fn setup_assert(&mut self) -> Result<(), TestError> {
        let db_provider = NetDbProvider::new_from_config(&self.configs).await;
        for db in db_provider.iter() {
            let num_irises = db.gpu_iris_store.count_irises().await?;
            assert_eq!(num_irises, self.params.max_indexation_id() as usize);

            let num_irises = db.cpu_iris_store.count_irises().await?;
            assert_eq!(num_irises, 0);

            let num_modifications = db.gpu_iris_store.last_modifications(1).await?;
            assert_eq!(num_modifications.len(), 0);

            let num_modifications = db.cpu_iris_store.last_modifications(1).await?;
            assert_eq!(num_modifications.len(), 0);
        }

        // Assert localstack.
        for config in self.configs.iter() {
            let aws_clients = iris_mpc::services::aws::clients::AwsClients::new(config).await?;
            let deletions = get_iris_deletions(
                config,
                &aws_clients.s3_client,
                self.params.max_indexation_id(),
            )
            .await?;
            assert_eq!(deletions.len(), 0);
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
