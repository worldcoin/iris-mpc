/// wal_108 — WAL materialisation from scratch: no prior checkpoint.
///
/// Verifies that `sidecar_main` builds a correct graph starting from an empty
/// state (no checkpoint row, no prior S3 object) by streaming all WAL mutations,
/// reaching 3-party BLAKE3 hash consensus, and writing the first checkpoint.
///
/// Using `run_hawk!` would only confirm the server started without
/// crashing; it would not verify that the WAL was correctly applied.  The sidecar
/// produces a checkpoint whose hash is compared against a reference materialised
/// in the test process, making the correctness of the WAL roll-forward observable.
///
/// This is distinct from:
///   wal_100 — empty WAL *and* no checkpoint (pure cold start)
///   wal_101 — WAL delta applied on top of an existing checkpoint
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_new_checkpoint,
        wal_builder::WalMutationBuilder,
    },
};

/// Number of WAL mutations to seed.  Large enough to exercise the roll-forward
/// path non-trivially; small enough to keep the test fast.
const WAL_MUTATION_COUNT: i64 = 20;
const EDGES_START_MOD_ID: i64 = WAL_MUTATION_COUNT + 1;
const TOTAL_MUTATIONS: usize = (WAL_MUTATION_COUNT as usize) * 2; // nodes + edges

pub struct Wal108 {
    nodes: Option<CpuNodes>,
}

impl Wal108 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal108 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // No checkpoint — sidecar will start from an empty graph.
        // Seed WAL mutations 1..=20 for all three parties.
        let builder = WalMutationBuilder::new()
            .add_nodes_sequential(WAL_MUTATION_COUNT as usize, 1)
            .add_edges_wrapping(WAL_MUTATION_COUNT as usize, EDGES_START_MOD_ID, 0);

        builder.build(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(TOTAL_MUTATIONS)
            .assert_max_modification_id(EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1)
            .assert_checkpoint_count(0);
        nodes
            .apply_uniform_assertions(&pre)
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
        // baseline = 0; wait for the sidecar to produce the first checkpoint.
        let res = wait_for_new_checkpoint(
            nodes,
            &ctx.configs,
            /* baseline */ 0,
            Duration::from_secs(120),
        )
        .await;
        stop_and_join!(shutdown, sidecar_set);
        res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Sidecar materialised the full WAL from scratch and wrote the first checkpoint.
        let last_mod_id = EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1;
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_MUTATIONS)
            .assert_max_modification_id(last_mod_id)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(last_mod_id)
            .assert_s3_object_exists(true);
        nodes
            .apply_uniform_assertions(&post)
            .await?;

        // All 3 parties must agree on the BLAKE3 hash and cross-check against the
        // reference hash computed from the full WAL in the test process — proves
        // the sidecar applied all mutations correctly.
        nodes.assert_consensus_and_reference().await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
