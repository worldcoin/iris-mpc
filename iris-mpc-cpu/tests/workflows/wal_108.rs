/// wal_108 — V3 graph load with pre-seeded WAL mutations, no prior checkpoint.
///
/// hawk_main must handle the case where WAL mutations already exist in the DB
/// but no checkpoint has ever been written.  It starts from an empty in-memory
/// graph (V3 path), streams all WAL mutations in order, applies them, and
/// signals ready.
///
/// This is distinct from:
///   wal_100 — empty WAL *and* no checkpoint (pure cold start)
///   wal_101 — WAL delta applied on top of an existing checkpoint
///
/// Termination condition: TC-1 (wait_for_all_ready)
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
    },
};

/// Number of WAL mutations to seed.  Large enough to exercise the roll-forward
/// path non-trivially; small enough to keep the test fast.
const WAL_MUTATION_COUNT: i64 = 20;
const EDGES_START_MOD_ID: i64 = WAL_MUTATION_COUNT + 1;
const TOTAL_MUTATIONS: usize = (WAL_MUTATION_COUNT as usize) * 2; // nodes + edges

pub struct Wal108 {
    nodes: Option<CpuNodes>,
}

impl Wal108 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal108 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // No checkpoint — hawk_main will start from an empty graph.
        // Seed WAL mutations 1..=20 for all three parties.
        let builder = (1i64..=WAL_MUTATION_COUNT).fold(WalMutationBuilder::new(), |b, id| {
            b.add_node(id, (id - 1) as u32, 1)
        });

        // Add edges: each node connects to the next two neighbors (wrapping).
        let num_nodes = WAL_MUTATION_COUNT as u32;
        let builder = (0..WAL_MUTATION_COUNT)
            .fold(builder, |b, idx| {
                let base = idx as u32;
                let neighbor1 = (base + 1) % num_nodes;
                let neighbor2 = (base + 2) % num_nodes;
                b.add_edges(EDGES_START_MOD_ID + idx, base, vec![neighbor1, neighbor2], 0)
            });

        builder.seed_all(&nodes).await?;

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
        let shutdown = CancellationToken::new();
        let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
        let ready_res =
            wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
        stop_and_join!(shutdown, hawk_set);
        ready_res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // hawk_main does not consume or checkpoint WAL entries; DB state unchanged.
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_MUTATIONS)
            .assert_max_modification_id(EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
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
