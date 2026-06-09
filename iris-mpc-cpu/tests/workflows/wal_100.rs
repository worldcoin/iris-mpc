/// wal_100 — hawk_main loads a V3 checkpoint on first run after genesis reset.
///
/// Simulates the scenario where a cluster has been reset to a genesis state and
/// the V3-format checkpoint is the only graph on disk.
///
/// Sequence:
///   1. Seed INITIAL_CHECKPOINT_NODES mutations into the WAL.
///   2. Create a V3-format checkpoint from those mutations (no `seq_no` field).
///      The WAL is intentionally not cleared — hawk_main will see WAL rows that
///      are already fully covered by the checkpoint, which is the normal state
///      after a genesis reset.
///   3. Add DELTA_NODES new mutations after the checkpoint to exercise the
///      roll-forward path (graph compaction pipeline + WAL write).
///   4. Run hawk_main; assert it reaches "ready" — confirming V3 load works.
///   5. sidecar cycle: rollthe DELTA_NODES into a GraphFormat::Current checkpoint
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
    },
    workflows::expect_sidecar_success,
};

/// Nodes seeded before the V3 checkpoint — fully covered by the checkpoint.
const INITIAL_CHECKPOINT_NODES: usize = 50;
/// Nodes added after the checkpoint — hawk_main must roll these forward on startup.
const DELTA_NODES: usize = 20;

#[derive(Default)]
pub struct Wal100 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal100 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal100 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?;
        let mut builder = WalMutationBuilder::new();

        // Phase 1: build the genesis graph and checkpoint it in V3 format.
        // The WAL is left intact — hawk_main will see rows already covered by
        // the checkpoint, which mirrors real genesis-reset behaviour.
        builder.add_nodes(INITIAL_CHECKPOINT_NODES);
        builder.build(&nodes).await?;
        nodes.make_checkpoints_v3().await?;

        // Phase 2: add a small delta that hawk_main must roll forward on startup
        // via the graph compaction pipeline, writing those mutations into the WAL.
        builder.add_nodes(DELTA_NODES);
        builder.build(&nodes).await?;

        self.nodes = Some(nodes);
        self.builder = Some(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let total = INITIAL_CHECKPOINT_NODES + DELTA_NODES;
        let pre = WalAssertions::new()
            .assert_wal_row_count(total)
            .assert_max_modification_id(total as i64)
            .assert_checkpoint_count(1)
            // Checkpoint was created after only the initial batch.
            .assert_latest_checkpoint_mod_id(INITIAL_CHECKPOINT_NODES as i64);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // hawk_main must load the V3 checkpoint, replay the DELTA_NODES WAL
        // mutations through the graph compaction pipeline, and reach ready.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            res.and(stop_and_join!(shutdown, hawk_set))?;
        }

        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await
        }
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let total = INITIAL_CHECKPOINT_NODES + DELTA_NODES;
        let post = WalAssertions::new()
            .assert_wal_row_count(total)
            .assert_max_modification_id(total as i64)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(total as i64);
        nodes.apply_uniform_assertions(&post).await?;
        nodes.assert_checkpoint_hashes_agree().await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
