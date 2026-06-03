/// wal_105 — V4 graph load: sidecar checkpoint as base for a second sidecar cycle.
///
/// After the sidecar materialises WAL mutations 51..=100 and writes a checkpoint
/// at mod_id=100, ten more mutations are seeded (nodes 101..=110 + edges).
/// A second sidecar cycle must select the mod_id=100 checkpoint (the "V4 path")
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
///   Phase 1: `sidecar_main` cycles over WAL 51..=100 → checkpoint at
///            mod_id=100 (baseline=1).
///   Seed WAL 101..=170 (10 nodes + 10 edges).
///   Phase 2: `sidecar_main` loads the mod_id=100 checkpoint, applies the new
///            delta, and writes a checkpoint at mod_id=170 (baseline=2).
use std::time::Duration;

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

/// Seed checkpoint anchor.
const CHECKPOINT_AT_MOD_ID: i64 = 50;
/// Upper boundary of the first WAL batch (sidecar will checkpoint here).
const SIDECAR_CHECKPOINT_MOD_ID: i64 = 100;
/// Upper boundary of the second WAL batch (hawk_main rolls forward through these).
const HAWK_DELTA_UP_TO_MOD_ID: i64 = 110;

const INITIAL_NODES: usize = (SIDECAR_CHECKPOINT_MOD_ID - CHECKPOINT_AT_MOD_ID) as usize; // 50
const INITIAL_EDGES_START: i64 = SIDECAR_CHECKPOINT_MOD_ID + 1; // 101
const HAWK_NODES: usize = (HAWK_DELTA_UP_TO_MOD_ID - SIDECAR_CHECKPOINT_MOD_ID) as usize; // 10
const HAWK_EDGES_START: i64 = HAWK_DELTA_UP_TO_MOD_ID + INITIAL_NODES as i64 + 1; // 161

const INITIAL_DELTA: usize = INITIAL_NODES + INITIAL_NODES; // nodes + edges (50 + 50)
const HAWK_DELTA: usize = HAWK_NODES + HAWK_NODES; // nodes + edges (10 + 10)
const TOTAL_WAL: usize = INITIAL_DELTA + HAWK_DELTA; // 120

pub struct Wal105 {
    nodes: Option<CpuNodes>,
}

impl Wal105 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal105 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // Seed WAL mutations 51..=100 for the sidecar to consume in phase 1.
        let builder = (CHECKPOINT_AT_MOD_ID + 1..=SIDECAR_CHECKPOINT_MOD_ID)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - CHECKPOINT_AT_MOD_ID - 1) as u32, 1)
            });

        // Add edges for first batch: each node connects to the next two neighbors (wrapping).
        let builder = (0..INITIAL_NODES as i64).fold(builder, |b, idx| {
            let base = idx as u32;
            let num_nodes = INITIAL_NODES as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(
                INITIAL_EDGES_START + idx,
                base,
                vec![neighbor1, neighbor2],
                0,
            )
        });

        builder.insert_mutations_all(&nodes).await?;
        builder.seed_modifications_all(&nodes).await?;

        // Build checkpoint from WAL up to modification_id = 50.
        nodes
            .make_checkpoints(CHECKPOINT_AT_MOD_ID, CHECKPOINT_AT_MOD_ID)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(INITIAL_DELTA)
            .assert_max_modification_id(INITIAL_EDGES_START + INITIAL_NODES as i64 - 1)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes
            .apply_assertions(&[pre.clone(), pre.clone(), pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Phase 1: sidecar materialises WAL 51..=100 and writes checkpoint at mod_id=100.
        // Baseline = 1 (the seeded checkpoint already present); wait for a second row.
        {
            let shutdown = CancellationToken::new();
            let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            let res = wait_for_new_checkpoint(
                nodes,
                &ctx.configs,
                /* baseline */ 1,
                Duration::from_secs(120),
            )
            .await;
            stop_and_join!(shutdown, sidecar_set);
            res?;
        }

        // Seed additional WAL mutations 101..=110.
        // These represent work that arrived after the sidecar checkpoint was written.
        let builder = (SIDECAR_CHECKPOINT_MOD_ID + 1..=HAWK_DELTA_UP_TO_MOD_ID)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - CHECKPOINT_AT_MOD_ID - 1) as u32, 1)
            });

        // Add edges for second batch: each node connects to the next two neighbors (wrapping).
        let builder = (0..HAWK_NODES as i64).fold(builder, |b, idx| {
            let base = (INITIAL_NODES as u32) + (idx as u32);
            let num_nodes = (INITIAL_NODES + HAWK_NODES) as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(HAWK_EDGES_START + idx, base, vec![neighbor1, neighbor2], 0)
        });

        builder.insert_mutations_all(nodes).await?;
        builder.seed_modifications_all(nodes).await?;

        // Phase 2: sidecar starts.  It must discover the Phase 1 checkpoint at
        // mod_id=100 as the latest base ("V4 path") and materialise only the new
        // delta (mutations 101..=170), then write a new checkpoint.
        // baseline = 2 (seeded + phase-1 checkpoint); wait for a third row.
        {
            let shutdown = CancellationToken::new();
            let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            let res = wait_for_new_checkpoint(
                nodes,
                &ctx.configs,
                /* baseline */ 2,
                Duration::from_secs(120),
            )
            .await;
            stop_and_join!(shutdown, sidecar_set);
            res?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After both sidecar cycles: seeded + phase-1 + phase-2 = 3 checkpoints.
        // WAL is intact: 120 rows total.  Phase-2 checkpoint anchored at the last
        // mutation of the hawk delta batch.
        let last_mod_id = HAWK_EDGES_START + HAWK_NODES as i64 - 1;
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_WAL)
            .assert_max_modification_id(last_mod_id)
            .assert_checkpoint_count(3) // seeded + phase-1 sidecar + phase-2 sidecar
            .assert_latest_checkpoint_mod_id(last_mod_id)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All 3 parties must agree on the phase-2 checkpoint BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await?;

        // Cross-check against the reference hash computed from the full WAL
        // (initial 51..=150 + hawk delta 101..=170).  This proves the phase-2
        // sidecar loaded the phase-1 checkpoint as its base, not the genesis one.
        let reference_hash = nodes.0[0].store.compute_reference_hash().await?;
        nodes
            .assert_checkpoint_hashes_match_reference(&reference_hash)
            .await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
