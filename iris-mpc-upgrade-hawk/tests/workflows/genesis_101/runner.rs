//! simulates a MPC node, complete with configuration (HAWK and Genesis) and database connections.
//! a simulation consists of 3 MPC nodes; a utility is included to construct all 3 nodes from a list of common::Config structs

use crate::utils::{
    constants::COUNT_OF_PARTIES,
    mpc_node::{MpcNode, MpcNodes},
    resources::{self},
    s3_deletions::{get_aws_clients, upload_iris_deletions},
    HawkConfigs, TestError, TestRun, TestRunContextInfo,
};
use eyre::Result;
use iris_mpc_cpu::genesis::{
    get_iris_deletions,
    plaintext::{GenesisArgs, GenesisState},
};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use itertools::izip;
use tokio::task::JoinSet;

pub struct Test {
    configs: HawkConfigs,

    genesis_args: GenesisArgs,

    rng_state: u64,

    genesis_outputs: Option<Vec<GenesisState>>,
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
            genesis_outputs: None,
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
        for (node, expected) in izip!(
            self.get_nodes().await,
            self.genesis_outputs.take().unwrap().into_iter()
        ) {
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

                node.assert_graphs_match(&expected).await;

                assert_eq!(100, node.get_last_indexed_iris_id().await);
                assert_eq!(0, node.get_last_indexed_modification_id().await);
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

        let mut genesis_outputs = vec![];
        while let Some(r) = join_set.join_next().await {
            genesis_outputs.push(r.unwrap());
        }
        self.genesis_outputs.replace(genesis_outputs);

        let config = &self.configs[0];
        let deleted_serial_ids = vec![];
        let aws_clients = get_aws_clients(config).await.unwrap();
        upload_iris_deletions(
            &deleted_serial_ids,
            &aws_clients.s3_client,
            &config.environment,
        )
        .await
        .unwrap();

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

                assert_eq!(0, node.get_last_indexed_iris_id().await);
                assert_eq!(0, node.get_last_indexed_modification_id().await);
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        // Assert localstack.

        let config = &self.configs[0];
        let max_indexation_id = self.genesis_args.max_indexation_id;
        let aws_clients = get_aws_clients(config).await.unwrap();
        let deletions = get_iris_deletions(config, &aws_clients.s3_client, max_indexation_id)
            .await
            .unwrap();
        assert_eq!(deletions.len(), 0);

        Ok(())
    }

    async fn teardown(&mut self) -> Result<(), TestError> {
        Ok(())
    }

    async fn teardown_assert(&mut self) -> Result<(), TestError> {
        Ok(())
    }
}
