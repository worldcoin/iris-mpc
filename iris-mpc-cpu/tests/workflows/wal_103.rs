/// wal_103 — Combined: startup roll-forward then sidecar cycle.
///
/// Phase 1: `hawk_main` starts with a base checkpoint at mod_id=50 and WAL
/// mutations 51..=100, applies the delta, and signals ready.
///
/// Phase 2: `sidecar_main` cycles over the full WAL (anchoring at mod_id=100),
/// uploads a new checkpoint to S3, and inserts a DB row.
///
/// This is the most representative end-to-end scenario.
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

const CHECKPOINT_AT_MOD_ID: i64 = 50;
const WAL_UP_TO_MOD_ID: i64 = 100;
const NODES_COUNT: usize = (WAL_UP_TO_MOD_ID - CHECKPOINT_AT_MOD_ID) as usize; // 50
const EDGES_START_MOD_ID: i64 = WAL_UP_TO_MOD_ID + 1;
const DELTA_SIZE: usize = NODES_COUNT + NODES_COUNT; // nodes + edges

pub struct Wal103 {
    nodes: Option<CpuNodes>,
}

impl Wal103 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal103 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // Base checkpoint at modification_id = 50.
        nodes
            .seed_all(CHECKPOINT_AT_MOD_ID, CHECKPOINT_AT_MOD_ID)
            .await?;

        // WAL delta 51..=100.
        let builder = (CHECKPOINT_AT_MOD_ID + 1..=WAL_UP_TO_MOD_ID)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - CHECKPOINT_AT_MOD_ID - 1) as u32, 1)
            });

        // Add edges: each node connects to the next two neighbors (wrapping).
        let builder = (0..NODES_COUNT as i64).fold(builder, |b, idx| {
            let base = idx as u32;
            let num_nodes = NODES_COUNT as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(
                EDGES_START_MOD_ID + idx,
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
            .assert_wal_row_count(DELTA_SIZE)
            .assert_max_modification_id(EDGES_START_MOD_ID + NODES_COUNT as i64 - 1)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes
            .apply_assertions(&[pre.clone(), pre.clone(), pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Phase 1: hawk_main roll-forward.
        // hawk_main and sidecar_main use different port sets so they can co-exist,
        // but they are run sequentially here (not simultaneously) to isolate phases.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints the current WAL state.
        // baseline = 1 (the seeded checkpoint); wait for a second row.
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

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // After the sidecar cycle, the new checkpoint should be anchored at the last edge mod_id.
        let post = WalAssertions::new()
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(EDGES_START_MOD_ID + NODES_COUNT as i64 - 1)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All 3 parties must agree on the BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await?;

        // Materialise the full WAL (mutations 51..=100) in the test process
        // and verify the hash matches the sidecar's stored checkpoint hash.
        let reference_hash = nodes.0[0].store.compute_reference_hash().await?;
        nodes
            .assert_checkpoint_hashes_match_reference(&reference_hash)
            .await?;

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
