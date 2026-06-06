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

    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let post = WalAssertions::new().assert_checkpoint_count(1);
        nodes.apply_uniform_assertions(&post).await?;

        nodes.assert_checkpoint_hashes_agree().await?;

        // Bucket is shared across all parties; filter by party_id prefix to count per-party S3 objects.
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
