use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, NUM_GPU_IRISES_INIT},
        modifications::ModificationInput,
        modifications::ModificationType::{Reauth, ResetUpdate},
        mpc_node::{DbAssertions, MpcNodes},
        HawkConfigs, TestRun, TestRunContextInfo,
    },
};
use eyre::Result;
use iris_mpc_cpu::genesis::BatchSizeConfig;
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use rand::Rng;
use tokio::task::JoinSet;

const NUM_ITERATIONS: usize = 3;
const BATCH_SIZE: usize = 16;
const CHAOS_MAX_INDEXATION_ID: u32 = 64;
/// Max per-batch delay ceiling (each node picks a random ceiling 0..=MAX_DELAY_MS
/// at the start of each iteration, then each commit sleeps 0..=ceiling).
const MAX_DELAY_MS: u64 = 2000;

const MODIFICATIONS: [ModificationInput; 2] = [
    ModificationInput::new(1, 5, ResetUpdate, true, true),
    ModificationInput::new(2, 15, Reauth, true, true),
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
}

impl TestRun for Test {
    async fn exec(&mut self) -> Result<()> {
        for iteration in 0..NUM_ITERATIONS {
            tracing::info!("=== Chaos iteration {}/{} ===", iteration + 1, NUM_ITERATIONS);

            // Clear CPU tables and GPU modifications, then re-insert modifications
            let nodes = MpcNodes::new(&self.configs).await;
            let mut setup_set = JoinSet::new();
            for node in nodes {
                setup_set.spawn(async move {
                    node.clear_cpu_tables().await.unwrap();
                    node.clear_gpu_modifications().await.unwrap();
                    node.apply_modifications(&[], &MODIFICATIONS).await.unwrap();
                });
            }
            setup_set.join_all().await;

            // Each node gets a different random delay ceiling
            let mut rng = rand::thread_rng();
            let mut configs = self.configs.clone();
            for config in configs.iter_mut() {
                let delay = rng.gen_range(0..=MAX_DELAY_MS);
                config.chaos_max_persistence_delay_ms = Some(delay);
                tracing::info!(
                    "  Node {} chaos delay ceiling: {}ms",
                    config.party_id,
                    delay
                );
            }

            // Spawn all 3 nodes
            let mut join_set = JoinSet::new();
            for config in configs.iter().cloned() {
                join_set.spawn(async move {
                    exec_genesis(
                        ExecutionArgs::new(
                            BatchSizeConfig::Static { size: BATCH_SIZE },
                            CHAOS_MAX_INDEXATION_ID,
                            false,
                        ),
                        config,
                    )
                    .await
                });
            }
            join_runners!(join_set);

            // Assert all nodes agree on state
            let cpu_asserts = DbAssertions::new()
                .assert_num_irises(CHAOS_MAX_INDEXATION_ID as usize)
                .assert_num_modifications(0)
                .assert_last_indexed_iris_id(CHAOS_MAX_INDEXATION_ID)
                .assert_last_indexed_modification_id(MODIFICATIONS.len() as i64);

            let gpu_asserts = DbAssertions::new()
                .assert_num_irises(NUM_GPU_IRISES_INIT)
                .assert_num_modifications(MODIFICATIONS.len());

            let nodes = MpcNodes::new(&self.configs).await;
            nodes.apply_assertions(gpu_asserts, cpu_asserts).await;

            tracing::info!(
                "=== Chaos iteration {}/{} PASSED ===",
                iteration + 1,
                NUM_ITERATIONS
            );
        }

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        // All assertions are done per-iteration in exec()
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
