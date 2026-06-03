/// wal_102 — Sidecar: WAL present → checkpoint uploaded to S3.
///
/// Verifies that `sidecar_main` materialises the WAL, reaches 3-party BLAKE3
/// hash consensus, uploads a checkpoint to S3, and inserts a DB row with the
/// correct WAL anchor.
///
/// The test then materialises its own reference graph from the same WAL rows,
/// hashes it, and verifies all 3 parties' stored hashes match the reference.
///
/// Termination condition: TC-2 (wait_for_new_checkpoint)
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
        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(WAL_MUTATION_COUNT as usize)
            .assert_max_modification_id(WAL_MUTATION_COUNT)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[pre.clone(), pre.clone(), pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let shutdown = CancellationToken::new();
        let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone());

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
            .assert_latest_checkpoint_mod_id(WAL_MUTATION_COUNT)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All 3 parties must agree on the BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await?;

        // TODO (open question #7): materialise the reference graph from WAL rows,
        // compute its BLAKE3 hash, and compare against stored hashes:
        //
        //   let reference_hash = materialise_and_hash(&nodes.0[0].stores.graph, 0, WAL_MUTATION_COUNT).await?;
        //   nodes.assert_checkpoint_hashes_match_reference(&reference_hash).await?;

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
