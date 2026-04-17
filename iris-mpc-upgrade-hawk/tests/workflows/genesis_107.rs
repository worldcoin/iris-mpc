use std::sync::Arc;

use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, DEFAULT_GENESIS_ARGS, MAX_INDEXATION_ID},
        mpc_node::{DbAssertions, MpcNode, MpcNodes},
        plaintext_genesis, HawkConfigs, TestRun, TestRunContextInfo,
    },
};
use eyre::Result;
use iris_mpc_cpu::genesis::plaintext::{run_plaintext_genesis, GenesisState};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

const NUM_EXTRA_IRISES: usize = 10;

pub struct Test {
    configs: HawkConfigs,
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
    // index 50 irises
    // then insert 10 additional irises into the CPU database and update persistent state
    // then run genesis again (which should trigger rollback functionality)
    async fn exec(&mut self) -> Result<()> {
        // Execute genesis - first run indexing up to 50
        let genesis_args = DEFAULT_GENESIS_ARGS;
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let mut args = genesis_args.clone();
            args.max_indexation_id = 50;
            join_set.spawn(async move {
                exec_genesis(ExecutionArgs::from_plaintext_args(args, false), config).await
            });
        }
        join_runners!(join_set);

        // Insert 10 additional irises into the CPU database and update persistent state
        // This simulates a scenario where the CPU database thinks it indexed more irises
        // than what's in the S3 checkpoint, which should trigger the rollback functionality
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move {
                node.insert_extra_irises_into_cpu_store(50, NUM_EXTRA_IRISES)
                    .await
            });
        }
        join_runners!(join_set);

        // Execute genesis - second run indexing up to 100
        // The rollback functionality should:
        // 1. Detect that the database has more irises (60) than the checkpoint (50)
        // 2. Delete extra irises from the CPU database
        // 3. Reset last_indexed_iris_id to 50
        // 4. Continue indexing normally from 51 to 100
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let mut args = genesis_args.clone();
            args.max_indexation_id = 100;
            join_set.spawn(async move {
                exec_genesis(ExecutionArgs::from_plaintext_args(args, false), config).await
            });
        }
        join_runners!(join_set);

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        // Run plaintext genesis to get expected state
        let mut state_0 = GenesisState::default();
        state_0.src_db.irises =
            plaintext_genesis::init_plaintext_irises_db(&genesis_runner::get_irises());
        state_0.config = plaintext_genesis::init_plaintext_config(&self.configs[0]);
        state_0.args = DEFAULT_GENESIS_ARGS;
        state_0.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(state_0)
            .await
            .expect("Stage 1 of plaintext genesis execution failed");

        // No modifications - just continue to 100
        state_1.args.max_indexation_id = 100;

        let expected = run_plaintext_genesis(state_1)
            .await
            .expect("Stage 2 of plaintext genesis execution failed");

        // Assert databases
        // After rollback, the CPU database should have 100 irises (the real ones, not the fake ones)
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_num_modifications(0);

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_vector_ids(plaintext_genesis::get_vector_ids(&expected.dst_db.irises))
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(100)
            .assert_last_indexed_modification_id(0);

        let nodes = MpcNodes::new(&self.configs).await;
        nodes.apply_assertions(gpu_asserts, cpu_asserts).await;
        nodes
            .assert_s3_checkpoint_graphs(&self.configs, &expected.dst_db.graphs)
            .await?;

        Ok(())
    }

    async fn setup(&mut self, _ctx: &TestRunContextInfo) -> Result<()> {
        let test_deletions = vec![];
        genesis_runner::base_genesis_e2e_init(&self.configs, test_deletions).await
    }

    async fn setup_assert(&mut self) -> Result<()> {
        genesis_runner::base_genesis_e2e_init_assertions(&self.configs, 0).await
    }

    async fn teardown(&mut self) -> Result<()> {
        let nodes = MpcNodes::new(&self.configs).await;
        nodes.cleanup_s3_checkpoints(&self.configs).await
    }
}
