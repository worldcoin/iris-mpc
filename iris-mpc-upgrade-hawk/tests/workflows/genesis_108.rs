use crate::{
    join_runners,
    utils::{
        genesis_runner::{self, DEFAULT_GENESIS_ARGS, NUM_GPU_IRISES_INIT},
        mpc_node::{DbAssertions, MpcNodes},
        plaintext_genesis, HawkConfigs, TestRun, TestRunContextInfo,
    },
};
use eyre::Result;
use iris_mpc_cpu::genesis::plaintext::{run_plaintext_genesis, GenesisState};
use iris_mpc_upgrade_hawk::genesis::{exec as exec_genesis, ExecutionArgs};
use tokio::task::JoinSet;

/// Final iris count after all three genesis runs (25 irises per run).
const FINAL_INDEXATION_ID: usize = 75;

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
    // Run genesis three times, indexing 25 irises per run.
    //
    // After the second run, party 0's genesis graph checkpoint is deleted from its
    // CPU-side `genesis_graph_checkpoint` table. This puts party 0 into a state where
    // it has indexed irises in its iris store but has no record of where the corresponding
    // HNSW graph was checkpointed on S3.
    //
    // The third genesis run must therefore exercise the graph-rollback path:
    //   - Parties 1 and 2 still have their checkpoint entries and agree on a hash.
    //   - Party 0 has the previous entry and contributes an old hash.
    //   - All parties have the old hash so they all roll back to that one.
    async fn exec(&mut self) -> Result<()> {
        let genesis_args = DEFAULT_GENESIS_ARGS;

        // Run 1: index irises 1-25.
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let mut args = genesis_args.clone();
            args.max_indexation_id = 25;
            join_set.spawn(async move {
                exec_genesis(ExecutionArgs::from_plaintext_args(args, false), config).await
            });
        }
        join_runners!(join_set);

        // Run 2: index irises 26-50.
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let mut args = genesis_args.clone();
            args.max_indexation_id = 50;
            join_set.spawn(async move {
                exec_genesis(ExecutionArgs::from_plaintext_args(args, false), config).await
            });
        }
        join_runners!(join_set);

        // Corrupt party 0's state: delete its genesis graph checkpoint from the CPU store.
        // Parties 1 and 2 are untouched - they retain their checkpoint entries pointing to
        // the S3 object that contains the HNSW graph for the 50-iris state.
        MpcNodes::new(&self.configs)
            .await
            .delete_checkpoint_for_party(0)
            .await?;

        // Run 3: index irises 51-75.
        // Party 0 must recover its graph from the S3 checkpoint held by parties 1 and 2
        // before it can continue indexing. The graph-rollback logic is what enables this.
        let mut join_set = JoinSet::new();
        for config in self.configs.iter().cloned() {
            let mut args = genesis_args.clone();
            args.max_indexation_id = FINAL_INDEXATION_ID as u32;
            join_set.spawn(async move {
                exec_genesis(ExecutionArgs::from_plaintext_args(args, false), config).await
            });
        }
        join_runners!(join_set);

        Ok(())
    }

    async fn exec_assert(&mut self) -> Result<()> {
        // Build the expected state by chaining three plaintext genesis runs that mirror
        // the three actual genesis runs above.  The plaintext model does not model the
        // checkpoint deletion because the deletion only affects *how* party 0 reaches the
        // correct state (via graph rollback), not *what* the correct state is.
        let mut state_0 = GenesisState::default();
        state_0.src_db.irises =
            plaintext_genesis::init_plaintext_irises_db(&genesis_runner::get_irises());
        state_0.config = plaintext_genesis::init_plaintext_config(&self.configs[0]);
        state_0.args = DEFAULT_GENESIS_ARGS;
        state_0.args.max_indexation_id = 25;

        let mut state_1 = run_plaintext_genesis(state_0)
            .await
            .expect("Stage 1 of plaintext genesis execution failed");
        state_1.args.max_indexation_id = 50;

        let mut state_2 = run_plaintext_genesis(state_1)
            .await
            .expect("Stage 2 of plaintext genesis execution failed");
        state_2.args.max_indexation_id = FINAL_INDEXATION_ID as u32;

        let expected = run_plaintext_genesis(state_2)
            .await
            .expect("Stage 3 of plaintext genesis execution failed");

        // GPU store is the source of truth for the raw iris shares; genesis only reads
        // from it, so its count remains at NUM_GPU_IRISES_INIT throughout.
        let gpu_asserts = DbAssertions::new()
            .assert_num_irises(NUM_GPU_IRISES_INIT)
            .assert_num_modifications(0);

        // CPU store should reflect exactly the 75 irises that were indexed across the
        // three runs, including party 0's graph-rollback recovery on run 3.
        let cpu_asserts = DbAssertions::new()
            .assert_num_irises(FINAL_INDEXATION_ID)
            .assert_vector_ids(plaintext_genesis::get_vector_ids(&expected.dst_db.irises))
            .assert_num_modifications(0)
            .assert_last_indexed_iris_id(FINAL_INDEXATION_ID as u32)
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
