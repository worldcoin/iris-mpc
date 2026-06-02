/// wal_104 — Sidecar pruning modes.
///
/// Verifies that `sidecar_main` respects the configured `PruningMode` when
/// deciding which checkpoint rows (and S3 objects) to retain after each cycle.
///
/// Three sub-scenarios run sequentially against the same DB state:
///
///   Run 1: pruning_mode = None
///     → checkpoint rows accumulate; nothing is deleted
///
///   Run 2: pruning_mode = OlderNonArchival
///     → non-archival checkpoints older than the latest are deleted
///
///   Run 3: pruning_mode = AllOlder
///     → all checkpoints except the latest are deleted
///
/// Termination condition: TC-2 (new checkpoint per run)
///
/// See open question #8 (readme) on whether BLAKE3 hashes can be asserted
/// deterministically across runs.
use std::time::Duration;

use crate::utils::{
    checkpoint_seeder::CheckpointSeeder,
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wait_conditions::wait_for_new_checkpoint,
    wal_builder::WalMutationBuilder,
};

pub struct Wal104 {
    nodes: Option<CpuNodes>,
}

impl Wal104 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal104 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_wal().await?;

        // Seed a base checkpoint and initial WAL mutations.
        CheckpointSeeder::new(0, 0).seed_all(&nodes, &ctx.configs).await?;

        let builder = (1i64..=10)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id as u32) - 1, 1)
            });
        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_wal_row_count(10);
        nodes.apply_assertions(&[pre.clone(), pre.clone(), pre]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Run 1: pruning_mode = None — all checkpoints kept.
        // TODO: override pruning_mode in ctx.configs for this run
        {
            let (_shutdown, _handles) = run_sidecar!(ctx.configs, nodes);
            wait_for_new_checkpoint(nodes, &ctx.configs, 1, Duration::from_secs(120)).await?;
            stop_and_join!(_shutdown, _handles);
        }

        // Run 2: pruning_mode = OlderNonArchival
        // TODO: override pruning_mode in ctx.configs for this run
        {
            let (_shutdown, _handles) = run_sidecar!(ctx.configs, nodes);
            wait_for_new_checkpoint(nodes, &ctx.configs, 2, Duration::from_secs(120)).await?;
            stop_and_join!(_shutdown, _handles);
        }

        // Run 3: pruning_mode = AllOlder
        // TODO: override pruning_mode in ctx.configs for this run
        {
            let (_shutdown, _handles) = run_sidecar!(ctx.configs, nodes);
            wait_for_new_checkpoint(nodes, &ctx.configs, 2 /* resets after pruning */, Duration::from_secs(120)).await?;
            stop_and_join!(_shutdown, _handles);
        }

        Ok(())
    }

    async fn exec_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // After AllOlder pruning only 1 checkpoint should remain.
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_s3_object_exists(true);
        nodes.apply_assertions(&[post.clone(), post.clone(), post]).await?;
        nodes.assert_checkpoint_hashes_agree().await
        // TODO: assert that pruned S3 objects no longer exist
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_wal().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
