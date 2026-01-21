use std::sync::Arc;

use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, default_genesis_args, MAX_INDEXATION_ID},
        modifications::{
            self, ModificationInput,
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

const MODIFICATIONS_START: [ModificationInput; 4] = [
    ModificationInput::new(1, 5, ResetUpdate, true, true),
    ModificationInput::new(2, 15, Uniqueness, true, true),
    ModificationInput::new(3, 25, Reauth, false, false),
    ModificationInput::new(4, 55, Uniqueness, false, false),
];

const MODIFICATIONS_END: [ModificationInput; 9] = [
    ModificationInput::new(1, 5, ResetUpdate, true, true),
    ModificationInput::new(2, 15, Uniqueness, true, true),
    ModificationInput::new(3, 25, Reauth, true, true),
    ModificationInput::new(4, 55, Uniqueness, true, true),
    ModificationInput::new(5, 60, ResetUpdate, true, true),
    ModificationInput::new(6, 70, Reauth, true, true),
    ModificationInput::new(7, 10, ResetUpdate, true, true),
    ModificationInput::new(8, 20, Reauth, true, true),
    ModificationInput::new(9, 30, ResetUpdate, false, false),
];

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
    async fn exec(&mut self) -> Result<()> {
        // Insert initial modifications
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set
                .spawn(async move { node.apply_modifications(&[], &MODIFICATIONS_START).await });
        }
        join_runners!(join_set);

        // Execute initial genesis run
        let genesis_args = default_genesis_args();
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let batch_size_config = genesis_args.batch_size_config.clone();
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        batch_size_config,
                        50,
                        false,
                        false,
                    ),
                    config,
                )
                .await
            });
        }
        join_runners!(join_set);

        // Persist initial modificatgions, and insert additional modifications
        let mut join_set = JoinSet::new();
        for node in self.get_nodes().await {
            join_set.spawn(async move {
                node.apply_modifications(&MODIFICATIONS_START, &MODIFICATIONS_END)
                    .await
            });
        }
        join_runners!(join_set);

        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let batch_size_config = genesis_args.batch_size_config.clone();
            join_set.spawn(async move {
                exec_genesis(
                    ExecutionArgs::new(
                        batch_size_config,
                        100,
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
        plaintext_genesis::apply_modifications(&mut state_0.src_db, &[], &MODIFICATIONS_START)?;
        state_0.args = default_genesis_args();
        state_0.args.max_indexation_id = 50;

        let mut state_1 = run_plaintext_genesis(state_0)
            .await
            .expect("Stage 1 of plaintext genesis execution failed");

        plaintext_genesis::apply_modifications(
            &mut state_1.src_db,
            &MODIFICATIONS_START,
            &MODIFICATIONS_END,
        )?;
        state_1.args.max_indexation_id = 100;

        let expected = run_plaintext_genesis(state_1)
            .await
            .expect("Stage 2 of plaintext genesis execution failed");

        // Number of modifications on already-indexed irises which are processed by genesis delta
        let num_updating_modifications = modifications::modifications_extension_updates(
            &MODIFICATIONS_START,
            &MODIFICATIONS_END,
        )
        .into_iter()
        .filter(|serial_id| *serial_id <= 50)
        .count();

        // Assert databases
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_num_modifications(MODIFICATIONS_END.len());

        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(MAX_INDEXATION_ID)
            .assert_vector_ids(plaintext_genesis::get_vector_ids(&expected.dst_db.irises))
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(100)
            .assert_last_indexed_modification_id(8)
            .assert_hnsw_layer_0_size(MAX_INDEXATION_ID + num_updating_modifications)
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
