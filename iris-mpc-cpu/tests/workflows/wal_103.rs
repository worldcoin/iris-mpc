/// wal_103 — Combined: startup roll-forward then sidecar cycle.
///
/// Phase 1: `hawk_main` starts with a base checkpoint at mod_id=50 and WAL
/// mutations 51..=100, applies the delta, and signals ready (TC-1).
///
/// Phase 2: `sidecar_main` cycles over the full WAL (anchoring at mod_id=100),
/// uploads a new checkpoint to S3, and inserts a DB row (TC-2).
///
/// This is the most representative integration scenario: it exercises the full
/// "startup roll-forward → live operation → checkpoint" pipeline.
use std::time::Duration;

use crate::utils::{
    checkpoint_seeder::CheckpointSeeder,
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wait_conditions::{wait_for_all_ready, wait_for_new_checkpoint},
    wal_builder::WalMutationBuilder,
};

const CHECKPOINT_AT_MOD_ID: i64 = 50;
const WAL_MUTATIONS_UP_TO: i64 = 100;

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
        nodes.truncate_wal().await?;

        // Base checkpoint at modification_id = 50.
        CheckpointSeeder::new(50, CHECKPOINT_AT_MOD_ID)
            .seed_all(&nodes, &ctx.configs)
            .await?;

        // WAL mutations 51..=100 (the delta hawk_main will roll forward).
        let builder = (CHECKPOINT_AT_MOD_ID + 1..=WAL_MUTATIONS_UP_TO)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id as u32) - 1, 1)
            });
        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let expected_wal_rows = (WAL_MUTATIONS_UP_TO - CHECKPOINT_AT_MOD_ID) as usize;
        let pre = WalAssertions::new()
            .assert_wal_row_count(expected_wal_rows)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_assertions(&[pre.clone(), pre.clone(), pre]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Phase 1: hawk_main roll-forward.
        let (_shutdown_hawk, _handles_hawk) = run_hawk!(ctx.configs);
        wait_for_all_ready(&ctx.configs, Duration::from_secs(60)).await?;
        stop_and_join!(_shutdown_hawk, _handles_hawk);

        // Phase 2: sidecar checkpoints the rolled-forward state.
        // baseline = 1 (the seeded checkpoint); wait for a second row.
        let (_shutdown_sidecar, _handles_sidecar) = run_sidecar!(ctx.configs, nodes);
        wait_for_new_checkpoint(nodes, &ctx.configs, 1, Duration::from_secs(120)).await?;
        stop_and_join!(_shutdown_sidecar, _handles_sidecar);

        Ok(())
    }

    async fn exec_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // After sidecar cycle, latest checkpoint should be anchored at mod_id=100.
        let post = WalAssertions::new()
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(WAL_MUTATIONS_UP_TO)
            .assert_s3_object_exists(true);
        nodes.apply_assertions(&[post.clone(), post.clone(), post]).await?;
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
