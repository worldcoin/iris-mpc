/// wal_102 — Sidecar: WAL present → checkpoint uploaded to S3.
///
/// Verifies that `sidecar_main` materialises the WAL, reaches 3-party BLAKE3
/// hash consensus, uploads a checkpoint to S3, and inserts a DB row with the
/// correct WAL anchor.
///
/// The test then materialises its own reference graph from the same WAL rows,
/// hashes it, and verifies all 3 parties' stored hashes match the reference.
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

const WAL_MUTATION_COUNT: usize = 10;

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
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // No base checkpoint — sidecar starts from scratch.
        // Seed AddNode mutations 1..=10.
        WalMutationBuilder::new()
            .add_nodes(WAL_MUTATION_COUNT)
            .build(&nodes)
            .await?;
        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(WAL_MUTATION_COUNT)
            .assert_max_modification_id(WAL_MUTATION_COUNT as _)
            .assert_checkpoint_count(0);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
        expect_sidecar_success(shutdown, sidecar_set).await
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(WAL_MUTATION_COUNT as _);
        nodes.apply_uniform_assertions(&post).await?;

        // Materialise the same WAL rows in the test process, hash the result,
        // and verify it matches every party's stored BLAKE3.
        nodes.assert_consensus_and_reference().await?;

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
