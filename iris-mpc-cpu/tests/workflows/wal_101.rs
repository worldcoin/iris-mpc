/// wal_101 — Roll-forward: checkpoint at M, WAL mutations M+1..=N.
///
/// Verifies that `hawk_main` loads the base checkpoint from S3, streams the WAL
/// mutations in range (M, N], applies them to the in-memory graph, and signals
/// ready.  The WAL rows remain intact after startup (roll-forward does not
/// consume them).
///
/// Termination condition: TC-1 (wait_for_all_ready)
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, stop_and_join,
    utils::{
        checkpoint_seeder::CheckpointSeeder,
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
        wal_builder::WalMutationBuilder,
    },
};

const CHECKPOINT_AT_MOD_ID: i64 = 50;
const WAL_UP_TO_MOD_ID: i64 = 100;
const DELTA_SIZE: usize = (WAL_UP_TO_MOD_ID - CHECKPOINT_AT_MOD_ID) as usize; // 50

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
        nodes.truncate_checkpoint_tables().await?;

        // Seed base checkpoint anchored at modification_id = 50.
        CheckpointSeeder::new(50, CHECKPOINT_AT_MOD_ID)
            .seed_all(&nodes, &ctx.configs)
            .await?;

        // Seed WAL mutations 51..=100 — the delta hawk_main will roll forward.
        let builder = (CHECKPOINT_AT_MOD_ID + 1..=WAL_UP_TO_MOD_ID).fold(
            WalMutationBuilder::new(),
            |b, id| b.add_node(id, (id - CHECKPOINT_AT_MOD_ID - 1) as u32, 1),
        );
        builder.seed_all(&nodes).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(DELTA_SIZE)
            .assert_max_modification_id(WAL_UP_TO_MOD_ID)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_assertions(&[pre.clone(), pre.clone(), pre]).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone());
        let ready_res = wait_for_all_ready(
            &ctx.configs,
            &mut hawk_set,
            Duration::from_secs(60),
        )
        .await;
        stop_and_join!(shutdown, hawk_set);
        ready_res
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // WAL rows are NOT consumed by roll-forward; checkpoint row count unchanged.
        let post = WalAssertions::new()
            .assert_wal_row_count(DELTA_SIZE)
            .assert_max_modification_id(WAL_UP_TO_MOD_ID)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(CHECKPOINT_AT_MOD_ID);
        nodes.apply_assertions(&[post.clone(), post.clone(), post]).await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
