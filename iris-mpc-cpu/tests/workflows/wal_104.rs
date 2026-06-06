/// wal_104 — Three sequential sidecar cycles exercise each `PruningMode` variant.
///
/// Setup seeds 1 archival + 2 regular checkpoints (archival inserted first so it
/// has the lowest row id).  Cycles run in order: None → OlderNonArchival → AllOlder.
use iris_mpc_cpu::graph_checkpoint::PruningMode;
use tokio_util::sync::CancellationToken;

use super::expect_sidecar_success;
use crate::{
    run_sidecar,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wal_builder::WalMutationBuilder,
        MIN_MUTATIONS_PER_SIDECAR_CYCLE,
    },
};

#[derive(Default)]
pub struct Wal104 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal104 {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Run one sidecar cycle with the given pruning mode and wait for all tasks to complete.
async fn run_cycle(ctx: &CpuTestContext, pruning_mode: PruningMode) -> eyre::Result<()> {
    // Build a per-cycle config override with the desired pruning mode.
    let mut configs = ctx.configs.clone();
    for cfg in configs.iter_mut() {
        cfg.sidecar.pruning_mode = pruning_mode;
    }

    let shutdown = CancellationToken::new();
    let sidecar_set = run_sidecar!(configs, shutdown.clone(), ctx);
    expect_sidecar_success(shutdown, sidecar_set).await
}

impl TestRun for Wal104 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_archival_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        self.nodes.replace(nodes);
        self.builder.replace(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_checkpoint_count(3)
            .assert_wal_row_count(4 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let builder = self.builder.as_mut().unwrap();

        // None: all 3 seeded + 1 new = 4 checkpoints; nothing deleted.
        run_cycle(ctx, PruningMode::None).await?;
        nodes.assert_checkpoint_count(4).await?;

        // Sidecar requires new mutations to trigger a cycle.
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // OlderNonArchival: removes the 3 non-archival older checkpoints; archival survives → 2 remain.
        run_cycle(ctx, PruningMode::OlderNonArchival).await?;
        nodes.assert_checkpoint_count(2).await?;

        // Sidecar requires new mutations to trigger a cycle.
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // AllOlder: only the latest checkpoint survives.
        run_cycle(ctx, PruningMode::AllOlder).await?;

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // assert_checkpoint_count already verifies the S3 object count per party matches.
        let post = WalAssertions::new().assert_checkpoint_count(1);
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
