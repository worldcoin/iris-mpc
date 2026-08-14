/// wal_102 — Sidecar produces a checkpoint from a fresh WAL with no base checkpoint.
use tokio_util::sync::CancellationToken;

use super::{expect_sidecar_success, run_sidecar};
use crate::utils::{
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wal_builder::WalMutationBuilder,
    MIN_MUTATIONS_PER_SIDECAR_CYCLE,
};

#[derive(Default)]
pub struct Wal102 {
    nodes: Option<CpuNodes>,
    /// Expected graph digest, captured from the builder because the cycle prunes
    /// the WAL it was derived from.
    reference_hash: Option<[u8; 32]>,
}

impl Wal102 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal102 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?;

        // No base checkpoint — sidecar starts from scratch.
        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        self.reference_hash = Some(builder.reference_hash()?);
        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_max_modification_id(MIN_MUTATIONS_PER_SIDECAR_CYCLE as _)
            .assert_checkpoint_count(0);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let sidecar_set = run_sidecar(&ctx.configs, shutdown.clone(), ctx);
        expect_sidecar_success(shutdown, sidecar_set).await
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // The new checkpoint is the only one, so it is also the oldest: it covers
        // the whole WAL, and the prune leaves just the anchor row at its height.
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(MIN_MUTATIONS_PER_SIDECAR_CYCLE as _)
            .assert_wal_row_count(1)
            .assert_min_modification_id(MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64);
        nodes.apply_uniform_assertions(&post).await?;

        nodes.assert_checkpoint_hashes_agree().await?;
        nodes
            .assert_checkpoint_hashes_match_reference(self.reference_hash.as_ref().unwrap())
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
