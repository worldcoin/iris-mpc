/// wal_111 — A single sidecar cycle with `PruningMode::Tiered` exercises the
/// count-based recent tier together with `keep_every_nth` thinning.
///
/// Setup seeds 10 regular checkpoints (all created "now", none backdated), then
/// appends a fresh batch of mutations so the cycle has work to do and produces
/// one more checkpoint.  With `delete_older_than_days = 30`,
/// `keep_recent_count = 4`, `keep_every_nth = 4` the tiers resolve as follows
/// (recency_rank counts newest-first, 0 = newest):
///
/// - Nothing is backdated, so every checkpoint is younger than
///   `delete_older_than_days` and the ancient (age-based) tier never fires.
/// - The 4 most recent checkpoints are always kept (recent tier,
///   recency_rank < 4).
/// - Among the older checkpoints, the sparse tier keeps only every 4th by
///   version age (version_age % 4 == 0) and deletes the rest.
use iris_mpc_cpu::graph_checkpoint::{PruningMode, TieredPruningConfig};
use tokio_util::sync::CancellationToken;

use super::{expect_sidecar_success, run_sidecar};
use crate::utils::{
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wal_builder::WalMutationBuilder,
    MIN_MUTATIONS_PER_SIDECAR_CYCLE,
};

const MIN: usize = MIN_MUTATIONS_PER_SIDECAR_CYCLE;

/// Tiered bounds chosen so the seeded checkpoints land one in each tier and
/// exercise the `keep_every_nth` thinning in the sparse tier.
const TIERED: TieredPruningConfig = TieredPruningConfig {
    delete_older_than_days: 30,
    keep_recent_count: 4,
    keep_every_nth: 4,
};

#[derive(Default)]
pub struct Wal111 {
    nodes: Option<CpuNodes>,
}

impl Wal111 {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Run one sidecar cycle in tiered-pruning mode and wait for all tasks to finish.
async fn run_tiered_cycle(ctx: &CpuTestContext, tiered: TieredPruningConfig) -> eyre::Result<()> {
    let mut configs = ctx.configs.clone();
    for cfg in configs.iter_mut() {
        cfg.sidecar.pruning_mode = PruningMode::Tiered;
        cfg.sidecar.tiered_pruning = tiered;
    }

    let shutdown = CancellationToken::new();
    let sidecar_set = run_sidecar(&configs, shutdown.clone(), ctx);
    expect_sidecar_success(shutdown, sidecar_set).await
}

impl TestRun for Wal111 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?;

        let mut builder = WalMutationBuilder::new();

        for _ in 0..10 {
            builder.add_nodes(MIN);
            builder.build(&nodes).await?;
            nodes.make_checkpoints().await?;
        }

        // Fresh mutations (>= min_mutations_per_cycle) so the sidecar has work
        // beyond the agreed base and produces a new checkpoint this cycle.
        builder.add_nodes(MIN);
        builder.build(&nodes).await?;

        self.nodes.replace(nodes);
        Ok(())
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        run_tiered_cycle(ctx, TIERED).await?;

        // The 4 most recent checkpoints (recent tier) survive, plus every 4th
        // older checkpoint (sparse tier); the remaining older ones are pruned.
        nodes.assert_checkpoint_count(9).await?;

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // assert_checkpoint_count already verifies the S3 object count per party.
        let post = WalAssertions::new().assert_checkpoint_count(4);
        nodes.apply_uniform_assertions(&post).await?;

        let mut checkpoints = nodes.0[0]
            .store
            .graph
            .get_genesis_graph_checkpoints_including_deleted()
            .await?;
        // Validate most recent 4 are not deleted
        // Sort checkpoints by id descending
        checkpoints.sort_by_key(|c| -c.id);
        for checkpoint in checkpoints.iter().take(4) {
            eyre::ensure!(
                !checkpoint.is_deleted,
                "checkpoint {:#?} should not be deleted",
                checkpoint
            );
        }
        // Validate each 4th is deleted from the oldest
        for (i, checkpoint) in checkpoints.iter().skip(4).rev().enumerate() {
            if i % 4 == 0 {
                eyre::ensure!(
                    checkpoint.is_deleted,
                    "checkpoint {:#?} should be deleted",
                    checkpoint
                );
            } else {
                eyre::ensure!(
                    !checkpoint.is_deleted,
                    "checkpoint {:#?} should not be deleted",
                    checkpoint
                );
            }
        }
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
