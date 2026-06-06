/// wal_109 — hawk_main syncs staggered per-party WAL state, then sidecar checkpoints.
///
/// Parties start with different WAL progress (0 / 5 / 10 rows) for the same 10
/// modifications; hawk_main transfers missing mutations to bring all parties in sync.
use std::time::Duration;

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

pub struct Wal109 {
    nodes: Option<CpuNodes>,
}

impl Wal109 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal109 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder0 = WalMutationBuilder::new();
        builder0.add_nodes(10);
        for idx in 1..=10 {
            builder0.set_persisted(idx, false);
        }
        builder0.build_single(&nodes.0[0], false, true).await?;

        let mut builder1 = WalMutationBuilder::new();
        builder1.add_nodes(5);
        builder1.build_single(&nodes.0[1], true, true).await?;
        builder1.add_nodes(5);
        for idx in 6..=10 {
            builder1.set_persisted(idx, false);
        }
        builder1.build_single(&nodes.0[1], false, true).await?;

        WalMutationBuilder::new()
            .add_nodes(10)
            .build_single(&nodes.0[2], true, true)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        nodes
            .apply_assertions(&[
                WalAssertions::new()
                    .assert_wal_row_count(0)
                    .assert_checkpoint_count(0),
                WalAssertions::new()
                    .assert_wal_row_count(5)
                    .assert_max_modification_id(5)
                    .assert_checkpoint_count(0),
                WalAssertions::new()
                    .assert_wal_row_count(10)
                    .assert_max_modification_id(10)
                    .assert_checkpoint_count(0),
            ])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main syncs all parties to the full mutation set.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints the synced WAL state.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let post = WalAssertions::new()
            .assert_wal_row_count(10)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(10);
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
