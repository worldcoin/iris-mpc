/// wal_100 — Baseline startup, empty WAL.
///
/// Verifies that `hawk_main` starts cleanly and all 3 parties signal ready when
/// the DB has no checkpoint and no WAL mutations.
///
/// Termination condition: TC-1 (wait_for_all_ready)
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
    },
};

pub struct Wal100 {
    nodes: Option<CpuNodes>,
}

impl Wal100 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal100 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;
        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let zero = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[zero.clone(), zero.clone(), zero])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
        let ready_res =
            wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
        stop_and_join!(shutdown, hawk_set);
        ready_res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let expected = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[expected.clone(), expected.clone(), expected])
            .await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
