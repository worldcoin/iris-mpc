/// wal_104 — Sidecar pruning modes.
///
/// Three sequential sidecar cycles exercise the three `PruningMode` variants.
/// Each cycle starts from the DB state left by the previous one.
///
/// Setup seeds three checkpoints: one archival (oldest) and two regular.
///  The archival checkpoint is inserted first
/// so it receives the lowest DB row id and is therefore "oldest".
///
/// Run 1: `PruningMode::None`
///   → all 3 seeded checkpoints + 1 new = 4 checkpoints; S3 objects accumulate
///
/// Run 2: `PruningMode::OlderNonArchival`
///   → the 2 regular seeded checkpoints and the run-1 checkpoint are deleted;
///     the archival and the run-2 checkpoint survive → 2 checkpoints remain
///
/// Run 3: `PruningMode::AllOlder`
///   → all checkpoints except the latest are deleted; only 1 row remains and the
///     S3 bucket shrinks from the 3 objects seeded in setup to 1
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

pub struct Wal104 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal104 {
    pub fn new() -> Self {
        Self {
            nodes: None,
            builder: None,
        }
    }
}

/// Run one sidecar cycle and wait for a new checkpoint, with the given pruning mode.
///
/// `baseline` is the number of checkpoint rows before this cycle starts.
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
        // one checkpoint

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;
        // 2 checkpoints

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;
        // 3 checkpoints

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        self.nodes.replace(nodes);
        self.builder.replace(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // 1 archival + 2 regular seeded checkpoints; 10 nodes + 10 edges in the WAL.
        let pre = WalAssertions::new()
            .assert_checkpoint_count(3)
            .assert_wal_row_count(4 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let builder = self.builder.as_mut().unwrap();

        // Run 1: no pruning — all 3 seeded checkpoints + 1 new = 4 checkpoints.
        run_cycle(ctx, PruningMode::None).await?;
        nodes.assert_checkpoint_count(4).await?;

        // need to add more nodes to make the sidecar run
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // Run 2: prune non-archival older checkpoints.
        // The 2 regular seeded checkpoints and the run-1 checkpoint are deleted.
        // The archival checkpoint (oldest) and the run-2 checkpoint survive → 2 remain.
        run_cycle(ctx, PruningMode::OlderNonArchival).await?;
        nodes.assert_checkpoint_count(2).await?;

        // need to add more nodes to make the sidecar run
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // Run 3: prune all older — only the latest checkpoint remains.
        run_cycle(ctx, PruningMode::AllOlder).await?;

        Ok(())
    }

    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After AllOlder pruning, exactly 1 checkpoint row remains per party.
        let post = WalAssertions::new().assert_checkpoint_count(1);
        nodes.apply_uniform_assertions(&post).await?;

        nodes.assert_checkpoint_hashes_agree().await?;

        // List S3 keys for each party's bucket and verify that only the latest
        // checkpoint remains for that party after AllOlder pruning.
        // The shared bucket contains checkpoints from all 3 parties, so we filter by party_id.
        for (node, config) in nodes.0.iter().zip(ctx.configs.iter()) {
            let all_keys = node.list_s3_keys(&config.checkpoint_bucket).await?;
            let party_id_str = format!("{}/", node.config.party_id);
            let party_keys: Vec<String> = all_keys
                .iter()
                .filter(|k| {
                    k.starts_with(&party_id_str)
                        || k.starts_with(&format!("genesis/{}/", node.config.party_id))
                })
                .cloned()
                .collect();
            tracing::info!(
                party_id = node.config.party_id,
                bucket = %config.checkpoint_bucket,
                party_keys = ?party_keys,
                "S3 keys after AllOlder pruning"
            );
            eyre::ensure!(
                party_keys.len() == 1,
                "party {}: expected 1 S3 object after AllOlder pruning, got {} — keys: {:?}",
                node.config.party_id,
                party_keys.len(),
                party_keys,
            );
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
