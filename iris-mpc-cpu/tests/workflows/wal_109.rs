/// wal_109 — hawk_main syncs staggered per-party WAL state, then sidecar checkpoints.
///
/// Setup: all parties first receive MIN mutations and a common base checkpoint is
/// created.  Then 2*MIN staggered mutations are added with different per-party WAL
/// progress (MIN / 2*MIN / 3*MIN persisted rows total); hawk_main transfers the
/// missing mutations to bring all parties in sync, after which the sidecar creates
/// a second checkpoint.
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::expect_sidecar_success;
use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
        MIN_MUTATIONS_PER_SIDECAR_CYCLE,
    },
};

#[derive(Default)]
pub struct Wal109 {
    nodes: Option<CpuNodes>,
}

impl Wal109 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal109 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;
        let aws_client = ctx.make_aws_client().await?;

        // Phase 1: all parties get MIN mutations persisted — establishes the common
        // base checkpoint required by the sidecar checkpoint protocol.
        let mut builder0 = WalMutationBuilder::new();
        let mut builder1 = WalMutationBuilder::new();
        let mut builder2 = WalMutationBuilder::new();

        builder0.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder1.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder2.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);

        builder0.build_single(&nodes.0[0], true, true).await?;
        builder1.build_single(&nodes.0[1], true, true).await?;
        builder2.build_single(&nodes.0[2], true, true).await?;

        nodes.make_checkpoints().await?;

        // Phase 2: staggered mutations — parties have different WAL progress.
        // Party 0: 2*MIN new mutations, all unpersisted (hawk_main must transfer all).
        builder0.add_nodes(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        for idx in MIN_MUTATIONS_PER_SIDECAR_CYCLE + 1..=3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE {
            builder0.set_persisted(idx as _, false);
        }
        builder0.build_single(&nodes.0[0], false, true).await?;

        // Party 1: first MIN new mutations persisted, second MIN unpersisted.
        builder1.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder1.build_single(&nodes.0[1], true, true).await?;
        builder1.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        for idx in 2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE + 1..=3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE {
            builder1.set_persisted(idx as _, false);
        }
        builder1.build_single(&nodes.0[1], false, true).await?;

        // Party 2: 2*MIN new mutations, all persisted.
        builder2.add_nodes(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder2.build_single(&nodes.0[2], true, true).await?;

        nodes
            .init_iris_shares(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE, &aws_client)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        nodes
            .apply_assertions(&[
                // Party 0: only the MIN base mutations are persisted; the 2*MIN staggered
                // mutations are unpersisted and absent from the WAL.
                WalAssertions::new()
                    .assert_wal_row_count(MIN_MUTATIONS_PER_SIDECAR_CYCLE)
                    .assert_max_modification_id(MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
                    .assert_checkpoint_count(1),
                // Party 1: MIN base + MIN from the first staggered batch.
                WalAssertions::new()
                    .assert_wal_row_count(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
                    .assert_max_modification_id(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
                    .assert_checkpoint_count(1),
                // Party 2: MIN base + all 2*MIN staggered mutations persisted.
                WalAssertions::new()
                    .assert_wal_row_count(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
                    .assert_max_modification_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
                    .assert_checkpoint_count(1),
            ])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main syncs all parties to the full mutation set.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar checkpoints the synced WAL state.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        let post = WalAssertions::new()
            .assert_wal_row_count(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64);
        nodes.apply_uniform_assertions(&post).await?;

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
