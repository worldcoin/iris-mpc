/// wal_107 — hawk_main syncs extra mutations on party 0 to parties 1 and 2 at startup.
///
/// Sidecar consensus after the sync proves the transferred mutations were persisted to DB,
/// not just applied in-memory.
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
        MIN_MUTATIONS_PER_SIDECAR_CYCLE,
    },
};

pub struct Wal107 {
    nodes: Option<CpuNodes>,
}

impl Wal107 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal107 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // Party 0 gets extra mutations that parties 1 and 2 don't have yet.
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build_single(&nodes.0[0], true, true).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let p0_pre = WalAssertions::new()
            .assert_wal_row_count(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_max_modification_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
            .assert_checkpoint_count(1);
        let p12_pre = WalAssertions::new()
            .assert_wal_row_count(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_max_modification_id(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
            .assert_checkpoint_count(1);
        nodes.apply_split_assertions(&p0_pre, &p12_pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main syncs mutations and signals ready.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints; consensus proves sync persisted to DB.
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
            .assert_latest_checkpoint_mod_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64);
        nodes.apply_uniform_assertions(&post).await?;

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
