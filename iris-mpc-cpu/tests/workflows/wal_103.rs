use std::time::Duration;

/// wal_103 — hawk_main roll-forward from a base checkpoint, then sidecar cycle.
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

        nodes.make_checkpoints().await?;

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
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints the current WAL state.
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
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(WAL_LEN as _);
        nodes.apply_uniform_assertions(&post).await?;

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
