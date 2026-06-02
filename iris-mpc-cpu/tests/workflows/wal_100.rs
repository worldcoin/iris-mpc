/// wal_100 — Baseline startup, empty WAL.
///
/// Verifies that `hawk_main` starts cleanly and signals ready when the DB has no
/// checkpoint and no WAL mutations.  Smoke test for the startup path.
///
/// Termination condition: TC-1 (ready endpoint)
use std::time::Duration;

use crate::utils::{
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wait_conditions::wait_for_all_ready,
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
        nodes.truncate_wal().await?;
        // No checkpoint to seed, no WAL mutations.
        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let zero = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0);
        nodes.apply_assertions(&[zero.clone(), zero.clone(), zero]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let (_shutdown, _handles) = run_hawk!(ctx.configs);
        wait_for_all_ready(&ctx.configs, Duration::from_secs(60)).await?;
        stop_and_join!(_shutdown, _handles);
        Ok(())
    }

    async fn exec_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let expected = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0);
        nodes.apply_assertions(&[expected.clone(), expected.clone(), expected]).await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_wal().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
