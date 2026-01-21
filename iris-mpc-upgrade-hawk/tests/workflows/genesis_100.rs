use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, default_genesis_args, MAX_INDEXATION_ID},
        mpc_node::{DbAssertions, MpcNodes},
        plaintext_genesis::PlaintextGenesis,
        HawkConfigs, TestRun, TestRunContextInfo,
    },
};
use eyre::Result;
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

pub struct Test {
    configs: HawkConfigs,
}

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
        let genesis_args = default_genesis_args();
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let args = genesis_args.clone();
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        args.batch_size_config,
                        args.max_indexation_id,
                        false,
                        false,
                    ),
                    config,
                )
                .await
            });
        }
        join_runners!(join_set);

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        // Run plaintext genesis
        let config = &self.configs[0];
        let plaintext_irises = genesis_runner::get_irises();
        let expected = PlaintextGenesis::new(default_genesis_args(), config, &plaintext_irises)
            .run()
            .await
            .unwrap();

        // Assert databases
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_num_modifications(0);

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
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
        genesis_runner::base_genesis_e2e_init_assertions(&self.configs, 0).await
    }
}
