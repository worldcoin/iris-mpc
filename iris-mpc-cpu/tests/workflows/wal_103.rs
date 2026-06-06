use std::time::Duration;

/// wal_103 — Combined: startup roll-forward then sidecar cycle.
///
/// Phase 1: `hawk_main` starts with a base checkpoint at mod_id=50 and WAL
/// mutations 51..=100, applies the delta, and signals ready.
///
/// Phase 2: `sidecar_main` cycles over the full WAL (anchoring at mod_id=100),
/// uploads a new checkpoint to S3, and inserts a DB row.
///
/// This is the most representative end-to-end scenario.
use tokio_util::sync::CancellationToken;

use super::expect_sidecar_success;
use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
    },
};

const WAL_LEN: usize = 100;
const CHECKPOINT_AT_MOD_ID: i64 = 50;

pub struct Wal103 {
    nodes: Option<CpuNodes>,
}

impl Wal103 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal103 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(WAL_LEN / 2);
        builder.build(&nodes).await?;

        // Build checkpoint from WAL up to modification_id = 50.
        nodes
            .make_checkpoints(CHECKPOINT_AT_MOD_ID, CHECKPOINT_AT_MOD_ID)
            .await?;

        // WAL delta 51..=100.
        builder.add_nodes(WAL_LEN / 2).build(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(WAL_LEN)
            .assert_max_modification_id(WAL_LEN as _)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main roll-forward.
        // hawk_main and sidecar_main use different port sets so they can co-exist,
        // but they are run sequentially here (not simultaneously) to isolate phases.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints the current WAL state.
        // baseline = 1 (the seeded checkpoint); wait for a second row.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After the sidecar cycle, the new checkpoint should be anchored at the last edge mod_id.
        let post = WalAssertions::new()
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(WAL_LEN as _);
        nodes.apply_uniform_assertions(&post).await?;

        // Materialise the full WAL (mutations 51..=100) in the test process
        // and verify the hash matches the sidecar's stored checkpoint hash.
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
