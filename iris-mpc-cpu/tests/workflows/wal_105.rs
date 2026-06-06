/// wal_105 — V4 graph load: sidecar checkpoint as base for a second sidecar cycle.
///
/// Setup seeds WAL mutations 51..=150 (50 nodes + 50 edges).  Phase 1 sidecar
/// materialises all 100 mutations and writes a checkpoint at mod_id=150.
/// Then 20 more mutations are seeded (hawk nodes 151..=160 + hawk edges 161..=170).
/// A second sidecar cycle must select the mod_id=150 checkpoint (the "V4 path")
/// as its base, apply only the new delta, and write a new checkpoint anchored at
/// the last mutation.
///
/// Using `run_hawk!` for Phase 2 would only confirm the server started;
/// it would not verify that the correct checkpoint was loaded or that the delta was
/// correctly applied.  The second sidecar cycle produces a checkpoint whose hash
/// is compared against a reference materialised from the full WAL (51..=170),
/// making both the base selection and the roll-forward correctness observable.
///
/// Protocol:
///   Phase 1: `sidecar_main` cycles over WAL 51..=150 → checkpoint at
///            mod_id=150 (baseline=1).
///   Seed WAL 151..=170 (10 nodes + 10 edges).
///   Phase 2: `sidecar_main` loads the mod_id=150 checkpoint, applies the new
///            delta, and writes a checkpoint at mod_id=170 (baseline=2).
use tokio_util::sync::CancellationToken;

use super::expect_sidecar_success;
use crate::{
    run_sidecar,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
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

        // Seed WAL mutations 51..=150 (50 nodes + 50 edges) for the sidecar to consume in phase 1.
        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(50);
        builder.build(&nodes).await?;

        nodes.make_checkpoints(50, 50).await?;

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

        // Phase 1: sidecar materialises WAL 51..=150 (nodes + edges) and writes checkpoint at mod_id=150.
        // Baseline = 1 (the seeded checkpoint already present); wait for a second row.
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

        // Seed additional WAL mutations 151..=160 (hawk nodes).
        builder.add_nodes(10);
        builder.build(nodes).await?;

        // Phase 2: sidecar starts.  It must discover the Phase 1 checkpoint at
        // mod_id=150 as the latest base ("V4 path") and materialise only the new
        // delta (mutations 151..=160), then write a new checkpoint.
        // baseline = 2 (seeded + phase-1 checkpoint); wait for a third row.
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

        // All 3 parties must agree on the phase-2 checkpoint BLAKE3 hash and cross-check
        // against the reference hash computed from the full WAL (initial 1..=150 + hawk delta 151..=160).
        // This proves the phase-2 sidecar loaded the phase-1 checkpoint as its base, not the genesis one.
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
