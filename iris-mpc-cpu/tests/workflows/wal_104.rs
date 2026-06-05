/// wal_104 — Sidecar pruning modes.
///
/// Three sequential sidecar cycles exercise the three `PruningMode` variants.
/// Each cycle starts from the DB state left by the previous one.
///
/// Setup seeds three checkpoints: one archival (oldest) and two regular, plus 20
/// WAL mutations (10 nodes + 10 edges).  The archival checkpoint is inserted first
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

use crate::{
    run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
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

    /// Run one sidecar cycle and wait for a new checkpoint, with the given pruning mode.
    ///
    /// `baseline` is the number of checkpoint rows before this cycle starts.
    async fn run_cycle(&self, ctx: &CpuTestContext, pruning_mode: PruningMode) -> eyre::Result<()> {
        // Build a per-cycle config override with the desired pruning mode.
        let mut configs = ctx.configs.clone();
        for cfg in configs.iter_mut() {
            cfg.sidecar.pruning_mode = pruning_mode.clone();
        }

        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(configs, shutdown.clone(), ctx);
        stop_and_join!(shutdown, sidecar_set);
        Ok(())
    }
}

impl TestRun for Wal104 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // 10 AddNode mutations (modification_ids 1–10).
        // 10 AddEdges mutations (modification_ids 11–20): each node connects to the
        // next two neighbors (wrapping).
        const NUM_NODES: usize = 10;
        const EDGES_START_MOD_ID: i64 = 11;
        let builder = WalMutationBuilder::new()
            .add_nodes_sequential_from(1, NUM_NODES)
            .add_edges_wrapping(NUM_NODES, EDGES_START_MOD_ID);

        builder.build(&nodes).await?;

        // Build checkpoints from WAL after mutations are seeded, so each snapshot
        // reflects the correct graph state.  Order preserved: archival oldest, then
        // two regular checkpoints at later modification ids.  The sidecar will see
        // WAL mutations after modification_id=10 (the latest checkpoint) and build
        // its first real checkpoint from mutations 11–20.
        nodes.seed_all_archival(0, 0).await?; // archival, last_mod_id=0  (oldest)
        nodes.make_checkpoints(0, 5).await?; // regular,  last_mod_id=5
        nodes.make_checkpoints(0, 10).await?; // regular,  last_mod_id=10 (latest)

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // 1 archival + 2 regular seeded checkpoints; 10 nodes + 10 edges in the WAL.
        let pre = WalAssertions::new()
            .assert_checkpoint_count(3)
            .assert_wal_row_count(20);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Run 1: no pruning — all 3 seeded checkpoints + 1 new = 4 checkpoints.
        self.run_cycle(ctx, PruningMode::None).await?;
        nodes.assert_checkpoint_count(4).await?;

        // Run 2: prune non-archival older checkpoints.
        // The 2 regular seeded checkpoints and the run-1 checkpoint are deleted.
        // The archival checkpoint (oldest) and the run-2 checkpoint survive → 2 remain.
        self.run_cycle(ctx, PruningMode::OlderNonArchival).await?;
        nodes.assert_checkpoint_count(2).await?;

        // Run 3: prune all older — only the latest checkpoint remains.
        self.run_cycle(ctx, PruningMode::AllOlder).await?;

        Ok(())
    }

    async fn exec_assert(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After AllOlder pruning, exactly 1 checkpoint row remains per party.
        let post = WalAssertions::new().assert_checkpoint_count(1);
        nodes.apply_uniform_assertions(&post).await?;

        nodes.assert_checkpoint_hashes_agree().await?;

        // List S3 keys for each party's bucket and verify that the count decreased
        // from the 3 objects seeded in setup to exactly 1 after AllOlder pruning.
        for (node, config) in nodes.0.iter().zip(ctx.configs.iter()) {
            let keys = node.list_s3_keys(&config.checkpoint_bucket).await?;
            tracing::info!(
                party_id = node.config.party_id,
                bucket = %config.checkpoint_bucket,
                ?keys,
                "S3 keys after AllOlder pruning"
            );
            eyre::ensure!(
                keys.len() == 1,
                "party {}: expected 1 S3 object after AllOlder pruning, got {} — keys: {:?}",
                node.config.party_id,
                keys.len(),
                keys,
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
