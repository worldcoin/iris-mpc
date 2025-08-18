use crate::utils::{
    genesis_runner,
    mpc_node::{DbAssertions, MpcNodes},
    plaintext_genesis::PlaintextGenesis,
    s3_deletions::get_aws_clients,
    HawkConfigs, TestRun, TestRunContextInfo,
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

impl Test {
    pub fn new() -> Self {
        Self {
            configs: genesis_runner::get_node_configs(),
        }
    }
}

impl TestRun for Test {
    async fn exec(&mut self) -> Result<()> {
        // Execute genesis
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let genesis_args = DEFAULT_GENESIS_ARGS;
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
        join_set.join_all().await;

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        // Run plaintext genesis
        let config = &self.configs[0];
        let plaintext_irises = genesis_runner::get_irises();
        let expected = PlaintextGenesis::new(DEFAULT_GENESIS_ARGS, config, &plaintext_irises)
            .run()
            .await
            .unwrap();

        // Assert databases
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(DEFAULT_GENESIS_ARGS.max_indexation_id as usize)
            .assert_num_modifications(0);

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(DEFAULT_GENESIS_ARGS.max_indexation_id as usize)
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(100)
            .assert_last_indexed_modification_id(0)
            .assert_hnsw_graphs(expected.dst_db.graphs);

        let nodes = MpcNodes::new(&self.configs).await;
        nodes.apply_assertions(gpu_asserts, cpu_asserts).await;

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        let test_deletions = vec![];
        genesis_runner::base_genesis_e2e_init(&self.configs, test_deletions).await
    }

    async fn setup_assert(&mut self) -> Result<()> {
        // Assert databases
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(DEFAULT_GENESIS_ARGS.max_indexation_id as usize)
            .assert_num_modifications(0);

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(0)
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(0)
            .assert_last_indexed_modification_id(0);

        let nodes = MpcNodes::new(&self.configs).await;
        nodes.apply_assertions(gpu_asserts, cpu_asserts).await;

        // Assert localstack
        let config = &self.configs[0];
        let aws_clients = get_aws_clients(config).await.unwrap();
        let deletions = get_iris_deletions(config, &aws_clients.s3_client, 100)
            .await
            .unwrap();
        assert_eq!(deletions.len(), 0);

        Ok(())
    }
}
