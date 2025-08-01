use super::params::Params;
use crate::utils::{
    constants::COUNT_OF_PARTIES, resources, s3_client, HawkConfigs, NetDbProvider, TestError,
    TestExecutionEnvironment, TestRun, TestRunContextInfo,
};
use eyre::Result;
use futures::future::join_all;
use iris_mpc_cpu::genesis::get_iris_deletions;
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};

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
        // Set node process futures.
        let node_futures: Vec<_> = self
            .configs
            .iter()
            .cloned()
            .map(|config| exec_genesis(self.genesis_args.clone(), config))
            .collect();

        join_all_and_report_errors!(node_futures, "failed to exec genesis");

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<(), TestError> {
        let db_provider = NetDbProvider::new_from_config(&self.configs).await;
        for db in db_provider.iter() {
            let num_irises = db.cpu_iris_store.count_irises().await?;
            assert_eq!(num_irises, self.params.max_indexation_id() as usize);
        }

        // Assert CPU dB tables: iris, hawk_graph_entry, hawk_graph_links, persistent_state
        // TODO

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

            // todo: make assertion about the graph table?
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
