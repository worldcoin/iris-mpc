/// wal_105 — Second sidecar cycle loads a prior checkpoint as its base and applies only the delta.
/// Also exercise applying WAL to existing checkpoint in server_main
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{expect_sidecar_success, run_sidecar, stop_and_join};
use crate::{
    run_hawk,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
    },
};

/// Nodes in the initial batch — seeded, then checkpointed before adding more.
const INITIAL_CHECKPOINT_NODES: usize = 50;
/// Additional nodes written after the initial checkpoint (phase-1 delta).
const FIRST_DELTA_NODES: usize = 100;
/// Additional nodes written after the phase-1 checkpoint (phase-2 delta).
const SECOND_DELTA_NODES: usize = 10;

#[derive(Default)]
pub struct Wal105 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal105 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal105 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(INITIAL_CHECKPOINT_NODES);
        builder.build(&nodes).await?;

        nodes.make_checkpoints().await?;

        builder.add_nodes(FIRST_DELTA_NODES);
        builder.build(&nodes).await?;

        self.nodes = Some(nodes);
        self.builder.replace(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let total_after_setup = INITIAL_CHECKPOINT_NODES + FIRST_DELTA_NODES;
        let pre = WalAssertions::new()
            .assert_wal_row_count(total_after_setup)
            .assert_max_modification_id(total_after_setup as i64)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(INITIAL_CHECKPOINT_NODES as i64);
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
            res.and(stop_and_join(shutdown, &mut hawk_set).await)?;
        }

        // Phase 1: sidecar materialises WAL delta and writes a checkpoint.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar(&ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        let total_phase1 = INITIAL_CHECKPOINT_NODES + FIRST_DELTA_NODES;
        let pre = WalAssertions::new()
            .assert_wal_row_count(total_phase1)
            .assert_max_modification_id(total_phase1 as i64)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(total_phase1 as i64);
        nodes.apply_uniform_assertions(&pre).await?;

        builder.add_nodes(SECOND_DELTA_NODES);
        builder.build(nodes).await?;

        // Phase 2: sidecar loads the phase-1 checkpoint as base and applies only the new delta.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar(&ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let total_final = INITIAL_CHECKPOINT_NODES + FIRST_DELTA_NODES + SECOND_DELTA_NODES;
        let post = WalAssertions::new()
            .assert_wal_row_count(total_final)
            .assert_max_modification_id(total_final as i64)
            .assert_checkpoint_count(3) // seeded + phase-1 sidecar + phase-2 sidecar
            .assert_latest_checkpoint_mod_id(total_final as i64);
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
