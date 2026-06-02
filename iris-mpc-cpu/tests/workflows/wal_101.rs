/// wal_101 — Roll-forward: checkpoint at M, WAL mutations M+1..N.
///
/// Verifies that `hawk_main` loads the base checkpoint from S3, streams WAL
/// mutations in range (M, N], applies them to the in-memory graph, and signals
/// ready.  The WAL rows remain in the DB after startup (roll-forward does not
/// consume them).
///
/// Termination condition: TC-1 (ready endpoint)
use std::time::Duration;

use crate::utils::{
    checkpoint_seeder::CheckpointSeeder,
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wait_conditions::wait_for_all_ready,
    wal_builder::WalMutationBuilder,
};

const CHECKPOINT_AT_MOD_ID: i64 = 50;
const WAL_MUTATIONS_UP_TO: i64 = 100;

pub struct Wal101 {
    nodes: Option<CpuNodes>,
}

impl Wal101 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal101 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_wal().await?;

        // Seed base checkpoint at modification_id = 50.
        CheckpointSeeder::new(50, CHECKPOINT_AT_MOD_ID)
            .seed_all(&nodes, &ctx.configs)
            .await?;

        // Seed WAL mutations 51..=100 — these will be rolled forward on startup.
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
        let expected_wal_rows = (WAL_MUTATIONS_UP_TO - CHECKPOINT_AT_MOD_ID) as usize; // 50
        let pre = WalAssertions::new()
            .assert_wal_row_count(expected_wal_rows)
            .assert_max_modification_id(WAL_MUTATIONS_UP_TO)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_assertions(&[pre.clone(), pre.clone(), pre]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let (_shutdown, _handles) = run_hawk!(ctx.configs);
        wait_for_all_ready(&ctx.configs, Duration::from_secs(60)).await?;
        stop_and_join!(_shutdown, _handles);
        Ok(())
    }

    async fn exec_assert(&self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let expected_wal_rows = (WAL_MUTATIONS_UP_TO - CHECKPOINT_AT_MOD_ID) as usize;
        // WAL rows are NOT consumed by roll-forward; checkpoint count unchanged.
        let post = WalAssertions::new()
            .assert_wal_row_count(expected_wal_rows)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_assertions(&[post.clone(), post.clone(), post]).await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_wal().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
