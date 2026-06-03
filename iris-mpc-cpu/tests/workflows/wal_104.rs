/// wal_104 — Sidecar pruning modes.
///
/// Three sequential sidecar cycles exercise the three `PruningMode` variants.
/// Each cycle starts from the DB state left by the previous one.
///
/// Run 1: `PruningMode::None`
///   → checkpoint rows accumulate; S3 objects for all prior checkpoints still exist
///
/// Run 2: `PruningMode::OlderNonArchival`
///   → non-archival checkpoints older than the latest are deleted from S3 and DB
///
/// Run 3: `PruningMode::AllOlder`
///   → all checkpoints except the latest are deleted; only 1 row remains
///
/// Termination condition: TC-2 (wait_for_new_checkpoint) per run.
use std::time::Duration;

use iris_mpc_cpu::graph_checkpoint::PruningMode;
use tokio_util::sync::CancellationToken;

use crate::{
    run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_new_checkpoint,
        wal_builder::WalMutationBuilder,
    },
};

pub struct Wal104 {
    nodes: Option<CpuNodes>,
}

impl Wal104 {
    pub fn new() -> Self {
        Self { nodes: None }
    }

    /// Run one sidecar cycle and wait for TC-2, with the given pruning mode.
    ///
    /// `baseline` is the number of checkpoint rows before this cycle starts.
    async fn run_cycle(
        &self,
        ctx: &CpuTestContext,
        pruning_mode: PruningMode,
        baseline: usize,
    ) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Build a per-cycle config override with the desired pruning mode.
        let mut configs = ctx.configs.clone();
        for cfg in configs.iter_mut() {
            cfg.sidecar.pruning_mode = pruning_mode.clone();
        }

        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(configs, shutdown.clone(), ctx);
        let res =
            wait_for_new_checkpoint(nodes, &ctx.configs, baseline, Duration::from_secs(120)).await;
        stop_and_join!(shutdown, sidecar_set);
        res
    }
}

impl TestRun for Wal104 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // Seed a base checkpoint (anchor = 0) and 10 WAL mutations.
        nodes.seed_all(0, 0).await?;

        let builder = (1i64..=10).fold(WalMutationBuilder::new(), |b, id| {
            b.add_node(id, (id - 1) as u32, 1)
        });

        // Add edges: each node connects to the next two neighbors (wrapping).
        const NUM_NODES: u32 = 10;
        const EDGES_START_MOD_ID: i64 = 11;
        let builder = (0..10i64)
            .fold(builder, |b, idx| {
                let base = idx as u32;
                let neighbor1 = (base + 1) % NUM_NODES;
                let neighbor2 = (base + 2) % NUM_NODES;
                b.add_edges(EDGES_START_MOD_ID + idx, base, vec![neighbor1, neighbor2], 0)
            });

        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_wal_row_count(20); // 10 nodes + 10 edges
        nodes
            .apply_assertions(&[pre.clone(), pre.clone(), pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Run 1: no pruning — both the seeded checkpoint and the new one remain.
        self.run_cycle(ctx, PruningMode::None, 1).await?;

        // Run 2: prune non-archival older checkpoints — seeded + run-1 checkpoints deleted.
        // The new checkpoint (from run 2) becomes the only one after pruning.
        self.run_cycle(ctx, PruningMode::OlderNonArchival, 2)
            .await?;

        // Run 3: prune all older — only the latest checkpoint remains.
        self.run_cycle(ctx, PruningMode::AllOlder, 2).await?;

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After AllOlder pruning, exactly 1 checkpoint row remains per party.
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        nodes.assert_checkpoint_hashes_agree().await?;

        // TODO: assert that the S3 objects for the two pruned checkpoints (from
        // run 1 and run 2) no longer exist — requires storing their s3_keys before
        // they are pruned.

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
