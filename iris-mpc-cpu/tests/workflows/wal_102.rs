/// wal_102 — Sidecar: WAL present → checkpoint uploaded to S3.
///
/// Verifies that `sidecar_main` materialises the WAL, reaches 3-party BLAKE3
/// hash consensus, uploads a checkpoint to S3, and inserts a DB row with the
/// correct WAL anchor.
///
/// `min_mutations_per_cycle` is set to 1 in the test config so that the sidecar
/// cycles immediately without needing a large number of seeded mutations
/// (see open question #7 in readme).
///
/// Termination condition: TC-2 (S3 checkpoint appearance)
use std::time::Duration;

use crate::utils::{
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wait_conditions::wait_for_new_checkpoint,
    wal_builder::WalMutationBuilder,
};

const WAL_MUTATION_COUNT: i64 = 30;

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
        nodes.truncate_wal().await?;

        // Seed WAL mutations 1..=30.  No base checkpoint — sidecar starts from scratch.
        let builder = (1..=WAL_MUTATION_COUNT)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id as u32) - 1, 1)
            });
        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(WAL_MUTATION_COUNT as usize)
            .assert_max_modification_id(WAL_MUTATION_COUNT)
            .assert_checkpoint_count(0);
        nodes.apply_assertions(&[pre.clone(), pre.clone(), pre]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let (_shutdown, _handles) = run_sidecar!(ctx.configs, nodes);
        // Wait for one new checkpoint to appear across all 3 parties.
        wait_for_new_checkpoint(nodes, &ctx.configs, 0, Duration::from_secs(120)).await?;
        stop_and_join!(_shutdown, _handles);
        Ok(())
    }

    async fn exec_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(WAL_MUTATION_COUNT)
            .assert_s3_object_exists(true);
        nodes.apply_assertions(&[post.clone(), post.clone(), post]).await?;
        // All 3 parties must agree on the BLAKE3 hash.
        nodes.assert_checkpoint_hashes_agree().await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_wal().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
