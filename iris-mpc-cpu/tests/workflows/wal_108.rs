/// wal_108 — WAL materialisation from scratch: no prior checkpoint.
///
/// Verifies that `sidecar_main` builds a correct graph starting from an empty
/// state (no checkpoint row, no prior S3 object) by streaming all WAL mutations,
/// reaching 3-party BLAKE3 hash consensus, and writing the first checkpoint.
///
/// Using `run_hawk!` would only confirm the server started without
/// crashing; it would not verify that the WAL was correctly applied.  The sidecar
/// produces a checkpoint whose hash is compared against a reference materialised
/// in the test process, making the correctness of the WAL roll-forward observable.
///
/// This is distinct from:
///   wal_100 — empty WAL *and* no checkpoint (pure cold start)
///   wal_101 — WAL delta applied on top of an existing checkpoint
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

        // No checkpoint — sidecar will start from an empty graph.
        // Seed WAL mutations 1..=20 for all three parties.
        let builder = (1i64..=WAL_MUTATION_COUNT).fold(WalMutationBuilder::new(), |b, id| {
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

        builder.insert_mutations_all(&nodes).await?;
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
        // baseline = 0; wait for the sidecar to produce the first checkpoint.
        let res = wait_for_new_checkpoint(
            nodes,
            &ctx.configs,
            /* baseline */ 0,
            Duration::from_secs(120),
        )
        .await;
        stop_and_join!(shutdown, sidecar_set);
        res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Sidecar materialised the full WAL from scratch and wrote the first checkpoint.
        let last_mod_id = EDGES_START_MOD_ID + WAL_MUTATION_COUNT - 1;
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_MUTATIONS)
            .assert_max_modification_id(last_mod_id)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(last_mod_id)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All 3 parties must agree on the BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await?;

        // Cross-check against the reference hash computed from the full WAL in the
        // test process — proves the sidecar applied all mutations correctly.
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
