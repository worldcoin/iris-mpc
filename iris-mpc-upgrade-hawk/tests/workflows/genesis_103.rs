use std::sync::Arc;

use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, DEFAULT_GENESIS_ARGS, MAX_INDEXATION_ID},
        modifications::{
            ModificationInput,
            ModificationType::{Reauth, ResetUpdate, Uniqueness},
        },
        mpc_node::{DbAssertions, MpcNode, MpcNodes},
        plaintext_genesis, HawkConfigs, TestRun, TestRunContextInfo,
    },
};
use eyre::Result;
use iris_mpc_cpu::genesis::plaintext::{run_plaintext_genesis, GenesisState};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

pub struct Test {
    configs: HawkConfigs,
}

const MODIFICATIONS: [ModificationInput; 3] = [
    ModificationInput::new(1, 5, ResetUpdate, true, true),
    ModificationInput::new(2, 15, Reauth, true, true),
    ModificationInput::new(3, 25, Uniqueness, true, true),
];

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
    async fn exec(&mut self) -> Result<()> {
        // Execute genesis
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move { node.apply_modifications(&[], &MODIFICATIONS).await });
        }
        join_runners!(join_set);

        let genesis_args = DEFAULT_GENESIS_ARGS;
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let args = genesis_args;
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
        let mut state_0 = GenesisState::default();
        state_0.src_db.irises =
            plaintext_genesis::init_plaintext_irises_db(&genesis_runner::get_irises());
        state_0.config = plaintext_genesis::init_plaintext_config(&self.configs[0]);
        plaintext_genesis::apply_modifications(&mut state_0.src_db, &[], &MODIFICATIONS)?;
        state_0.args = DEFAULT_GENESIS_ARGS;

        let expected = run_plaintext_genesis(state_0)
            .await
            .expect("Plaintext genesis execution failed");

        // Assert databases
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_num_modifications(MODIFICATIONS.len());

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_vector_ids(plaintext_genesis::get_vector_ids(&expected.dst_db.irises))
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(100)
            .assert_last_indexed_modification_id(2)
            .assert_hnsw_layer_0_size(MAX_INDEXATION_ID)
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
