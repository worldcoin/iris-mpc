/// wal_102 — Sidecar: WAL present → checkpoint uploaded to S3.
///
/// Verifies that `sidecar_main` materialises the WAL, reaches 3-party BLAKE3
/// hash consensus, uploads a checkpoint to S3, and inserts a DB row with the
/// correct WAL anchor.
///
/// The test then materialises its own reference graph from the same WAL rows,
/// hashes it, and verifies all 3 parties' stored hashes match the reference.
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

/// Seed 10 AddNode mutations.  min_mutations_per_cycle = 5, so the sidecar
/// will cycle on the first pass.
const WAL_MUTATION_COUNT: i64 = 10;
const EDGES_START_MOD_ID: i64 = WAL_MUTATION_COUNT + 1;
const TOTAL_MUTATIONS: usize = (WAL_MUTATION_COUNT as usize) * 2; // nodes + edges

pub struct Wal102 {
    nodes: Option<CpuNodes>,
}

impl Wal102 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal102 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // No base checkpoint — sidecar starts from scratch.
        // Seed AddNode mutations 1..=10.
        let builder = (1..=WAL_MUTATION_COUNT).fold(WalMutationBuilder::new(), |b, id| {
            b.add_node(id, (id - 1) as u32, 1)
        });

        // Add edges: each node connects to the next two neighbors (wrapping).
        let num_nodes = WAL_MUTATION_COUNT as u32;
        let builder = (0..WAL_MUTATION_COUNT).fold(builder, |b, idx| {
            let base = idx as u32;
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
            .assert_wal_row_count(TOTAL_MUTATIONS)
            .assert_max_modification_id(EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[pre.clone(), pre.clone(), pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);

        let checkpoint_res = wait_for_new_checkpoint(
            nodes,
            &ctx.configs,
            /* baseline */ 0,
            Duration::from_secs(120),
        )
        .await;

        stop_and_join!(shutdown, sidecar_set);
        checkpoint_res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All 3 parties must agree on the BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await?;

        // Materialise the same WAL rows in the test process, hash the result,
        // and verify it matches every party's stored BLAKE3.
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
