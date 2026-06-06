use std::time::Duration;

/// wal_105 — Second sidecar cycle loads a prior checkpoint as its base and applies only the delta.
/// Also exercise applying WAL to existing checkpoint in server_main
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

pub struct Wal105 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal105 {
    pub fn new() -> Self {
        Self {
            nodes: None,
            builder: None,
        }
    }
}

impl TestRun for Wal105 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(50);
        builder.build(&nodes).await?;

        nodes.make_checkpoints().await?;

        builder.add_nodes(100);
        builder.build(&nodes).await?;

        self.nodes = Some(nodes);
        self.builder.replace(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(150)
            .assert_max_modification_id(150)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(50);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let builder = self.builder.as_mut().unwrap();

        // Phase 0: exercise server_main()
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 1: sidecar materialises WAL delta and writes a checkpoint.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        let pre = WalAssertions::new()
            .assert_wal_row_count(150)
            .assert_max_modification_id(150)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(150);
        nodes.apply_uniform_assertions(&pre).await?;

        builder.add_nodes(10);
        builder.build(nodes).await?;

        // Phase 2: sidecar loads the phase-1 checkpoint as base and applies only the new delta.
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
            .assert_wal_row_count(160)
            .assert_max_modification_id(160)
            .assert_checkpoint_count(3) // seeded + phase-1 sidecar + phase-2 sidecar
            .assert_latest_checkpoint_mod_id(160);
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
