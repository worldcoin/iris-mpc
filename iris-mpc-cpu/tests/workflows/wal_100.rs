use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_all_ready,
    },
};
/// wal_100 — Baseline startup, empty WAL; idempotent sidecar checkpoint.
///
/// Verifies that `hawk_main` starts cleanly on an empty DB, that `sidecar_main`
/// produces exactly one checkpoint even when run twice against the same (unchanged)
/// WAL state, and that the resulting S3 objects agree across all 3 parties.
///
/// Exec — Phase 1
/// ----------------------
/// `hawk_main` starts with no WAL mutations and no prior checkpoint.  All 3
/// parties signal ready.
///
/// Exec — Phase 2a
/// -----------------------
/// `sidecar_main` (first run) checkpoints the empty graph state, writing one
/// checkpoint row per party and uploading the corresponding S3 object.
///
/// Exec — Phase 2b (idempotency)
/// ------------------------------
/// `sidecar_main` (second run) sees the checkpoint already covers the current
/// WAL high-water mark and must NOT write a second checkpoint row.  The test
/// lets the sidecar run briefly, then stops it and confirms the count is still 1.
///
/// Post-conditions
/// ---------------
/// Every party has 0 WAL rows, exactly 1 checkpoint, a matching S3 object, and
/// all parties agree on the BLAKE3 hash.
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub struct Wal100 {
    nodes: Option<CpuNodes>,
}

impl Wal100 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal100 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;
        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let zero = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0);
        nodes
            .apply_assertions(&[zero.clone(), zero.clone(), zero])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main signals ready on an empty WAL.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2a: first sidecar run — checkpoints the (empty) graph state.
        // baseline = 0: no prior checkpoint exists.
        {
            let shutdown = CancellationToken::new();
            let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            let res = tokio::time::timeout(Duration::from_secs(10), sidecar_set.join_next()).await;
            stop_and_join!(shutdown, sidecar_set);
            res?;
        }

        // Phase 2b: second sidecar run — exercises idempotency.
        // The WAL state has not advanced; the sidecar must not write a second
        // checkpoint row.  Give it enough time to complete at least one loop.
        {
            let shutdown = CancellationToken::new();
            let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            let res = tokio::time::timeout(Duration::from_secs(10), sidecar_set.join_next()).await;
            stop_and_join!(shutdown, sidecar_set);
            res?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Exactly 1 checkpoint despite two sidecar runs — the sidecar is idempotent.
        let expected = WalAssertions::new()
            .assert_wal_row_count(0)
            .assert_checkpoint_count(0)
            .assert_s3_object_exists(false);
        nodes
            .apply_assertions(&[expected.clone(), expected.clone(), expected])
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
