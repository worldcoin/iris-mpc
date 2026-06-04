/// wal_101 — Roll-forward: checkpoint at M, WAL mutations M+1..=N.
///
/// Verifies that `sidecar_main` loads the base checkpoint from S3, materialises
/// the WAL mutations in range (M, N], reaches 3-party BLAKE3 hash consensus, and
/// writes a new checkpoint to S3 anchored at the last WAL mutation.
///
/// Using `run_hawk!` here would only confirm the server did not crash;
/// it would not verify that the WAL delta was correctly applied.  The sidecar
/// writes a concrete checkpoint whose hash can be compared against a reference
/// materialised in the test process, making the roll-forward correctness
/// observable.
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

const CHECKPOINT_AT_MOD_ID: i64 = 50;
const WAL_UP_TO_MOD_ID: i64 = 100;
const NODES_COUNT: usize = (WAL_UP_TO_MOD_ID - CHECKPOINT_AT_MOD_ID) as usize; // 50
const EDGES_START_MOD_ID: i64 = WAL_UP_TO_MOD_ID + 1;
const DELTA_SIZE: usize = NODES_COUNT + NODES_COUNT; // nodes + edges

pub struct Wal101 {
    nodes: Option<CpuNodes>,
}

impl Wal101 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal101 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;
        // Seed WAL mutations 51..=100 — the delta the sidecar will roll forward.
        let builder = WalMutationBuilder::new()
            .add_nodes_sequential_from(CHECKPOINT_AT_MOD_ID + 1, NODES_COUNT, 1)
            .add_edges_wrapping(NODES_COUNT, EDGES_START_MOD_ID, 0);

        // Insert mutations and seed a `modifications` row for every WAL entry so that
        // hawk_main's modification sync operates on a realistic, non-empty modifications table.
        builder.build(&nodes).await?;

        // Build checkpoint from WAL up to modification_id = 50.
        nodes
            .make_checkpoints(CHECKPOINT_AT_MOD_ID, CHECKPOINT_AT_MOD_ID)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(DELTA_SIZE)
            .assert_max_modification_id(EDGES_START_MOD_ID + NODES_COUNT as i64 - 1)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
        // baseline = 1 (the seeded checkpoint); wait for the sidecar's new row.
        let res = wait_for_new_checkpoint(
            nodes,
            &ctx.configs,
            /* baseline */ 1,
            Duration::from_secs(120),
        )
        .await;
        stop_and_join!(shutdown, sidecar_set);
        res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Sidecar applied WAL 51..=150 on top of the seeded checkpoint and wrote a new one.
        // WAL rows are not consumed; the seeded row remains alongside the new one.
        let last_mod_id = EDGES_START_MOD_ID + NODES_COUNT as i64 - 1;
        let post = WalAssertions::new()
            .assert_wal_row_count(DELTA_SIZE)
            .assert_max_modification_id(last_mod_id)
            .assert_checkpoint_count(2) // seeded + sidecar
            .assert_latest_checkpoint_mod_id(last_mod_id)
            .assert_s3_object_exists(true);
        nodes.apply_uniform_assertions(&post).await?;

        // Materialise the same WAL rows in the test process and verify the hash matches
        // every party's stored BLAKE3 — proving the roll-forward was correct.
        nodes.assert_consensus_and_reference().await?;

        Ok(())
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
