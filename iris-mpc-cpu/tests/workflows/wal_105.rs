/// wal_105 — V4 graph load: hawk_main after a sidecar checkpoint.
///
/// After the sidecar materialises WAL mutations 51..=100 and writes a
/// checkpoint at mod_id=100, ten more mutations (101..=110) are seeded.
/// hawk_main must select the sidecar checkpoint (the "V4 path") as the base,
/// apply only the 10-entry delta, and signal ready.
///
/// Protocol:
///   Phase 1: `sidecar_main` cycles over WAL 51..=100 → checkpoint at
///            mod_id=100 (TC-2, baseline=1).
///   Seed WAL 101..=110.
///   Phase 2: `hawk_main` loads the mod_id=100 checkpoint and rolls forward
///            mutations 101..=110 → signals ready (TC-1).
///
/// Termination conditions: TC-2 then TC-1.
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::{wait_for_all_ready, wait_for_new_checkpoint},
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

        // Seed genesis checkpoint at modification_id = 50.
        nodes
            .seed_all(CHECKPOINT_AT_MOD_ID, CHECKPOINT_AT_MOD_ID)
            .await?;

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

        builder.seed_all(&nodes).await?;
        builder.seed_modifications_all(&nodes).await?;

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

        builder.seed_all(nodes).await?;
        builder.seed_modifications_all(nodes).await?;

        // Phase 2: hawk_main starts.  It should discover the sidecar's checkpoint at
        // mod_id=100 as the latest base ("V4 path") and roll forward only mutations
        // 101..=110 — not the full WAL from mod_id=50.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // hawk_main does not create checkpoint rows; still 2 (seeded + sidecar).
        // WAL is intact: 120 rows total (60 nodes + 60 edges), max_modification_id = HAWK_EDGES_START + HAWK_NODES - 1.
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_WAL)
            .assert_max_modification_id(HAWK_EDGES_START + HAWK_NODES as i64 - 1)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(SIDECAR_CHECKPOINT_MOD_ID)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // The sidecar checkpoint hashes must still agree across parties.
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
