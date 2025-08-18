use std::sync::Arc;

use crate::utils::{
    genesis_runner, irises,
    mpc_node::{MpcNode, MpcNodes},
    plaintext_genesis::PlaintextGenesis,
    resources::{self},
    s3_deletions::get_aws_clients,
    HawkConfigs, IrisCodePair, TestRun, TestRunContextInfo,
};
use eyre::Result;
use iris_mpc_cpu::genesis::{get_iris_deletions, plaintext::GenesisArgs};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

pub struct Test {
    configs: HawkConfigs,
}

const DEFAULT_GENESIS_ARGS: GenesisArgs = GenesisArgs {
    max_indexation_id: 100,
    batch_size: 1,
    batch_size_error_rate: 250,
};

fn get_irises() -> Vec<IrisCodePair> {
    let irises_path =
        resources::get_resource_path("iris-shares-plaintext/20250710-synthetic-irises-1k.ndjson");
    irises::read_irises_from_ndjson(irises_path, 100).unwrap()
}

impl Test {
    pub fn new() -> Self {
        Self {
            configs: genesis_runner::get_node_configs(),
        }
    }

    async fn get_nodes(&self) -> impl Iterator<Item = Arc<MpcNode>> {
        MpcNodes::new(&self.configs).await.into_iter()
    }
}

impl TestRun for Test {
    // run genesis twice - first indexing 50 and then up to 100
    async fn exec(&mut self) -> Result<()> {
        // these need to be on separate tasks
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let genesis_args = DEFAULT_GENESIS_ARGS;
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        genesis_args.batch_size,
                        genesis_args.batch_size_error_rate,
                        50,
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

        // hack to get around an "address already in use" error, emitted by HawkHandle
        let service_ports = vec!["4003".into(), "4004".into(), "4005".into()];

        let mut join_set = JoinSet::new();
        for mut config in self.configs.iter().cloned() {
            let genesis_args = DEFAULT_GENESIS_ARGS;
            config.service_ports = service_ports.clone();
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        genesis_args.batch_size,
                        genesis_args.batch_size_error_rate,
                        100,
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

    async fn exec_assert(&mut self) -> Result<()> {
        let config = &self.configs[0];
        let plaintext_irises = get_irises();
        let expected = Arc::new(
            PlaintextGenesis::new(DEFAULT_GENESIS_ARGS, config, &plaintext_irises)
                .run()
                .await
                .unwrap(),
        );

        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            let expected = expected.clone();
            let max_indexation_id = DEFAULT_GENESIS_ARGS.max_indexation_id as usize;
            join_set.spawn(async move {
                // inspect the postgres tables - iris counts, modifications, and persisted_state
                let num_irises = node.gpu_stores.iris.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_stores.iris.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_modifications = node.gpu_stores.iris.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = node.cpu_stores.iris.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                assert_eq!(100, node.get_last_indexed_iris_id().await);
                assert_eq!(0, node.get_last_indexed_modification_id().await);

                node.assert_graphs_match(&expected).await;
            });
        }

        while let Some(r) = join_set.join_next().await {
            r.unwrap();
        }

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        let test_deletions = vec![];
        genesis_runner::base_genesis_e2e_init(&self.configs, test_deletions).await
    }

    async fn setup_assert(&mut self) -> Result<()> {
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            let genesis_args = DEFAULT_GENESIS_ARGS;
            let max_indexation_id = genesis_args.max_indexation_id as usize;
            join_set.spawn(async move {
                let num_irises = node.gpu_stores.iris.count_irises().await.unwrap();
                assert_eq!(num_irises, max_indexation_id);

                let num_irises = node.cpu_stores.iris.count_irises().await.unwrap();
                assert_eq!(num_irises, 0);

                let num_modifications = node.gpu_stores.iris.last_modifications(1).await.unwrap();
                assert_eq!(num_modifications.len(), 0);

                let num_modifications = node.cpu_stores.iris.last_modifications(1).await.unwrap();
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
        let aws_clients = get_aws_clients(config).await.unwrap();
        let deletions = get_iris_deletions(config, &aws_clients.s3_client, 100)
            .await
            .unwrap();
        assert_eq!(deletions.len(), 0);

        Ok(())
    }
}
