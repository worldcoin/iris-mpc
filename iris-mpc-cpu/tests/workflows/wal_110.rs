/// wal_110 — Modification sync: conflict detection (mutation mismatch).
///
/// Mirrors `test_hawk_sync_mutation_mismatch` from
/// `iris-mpc-upgrade-hawk/tests/e2e_hawk.rs`.
///
/// Setup
/// -----
///   Party 0 – modification NOT persisted; no WAL entry.
///             Will attempt to roll forward and request the bytes from party 1/2.
///   Party 1 – modification persisted; WAL bytes = AddNode { serial_id: 1 }.
///   Party 2 – modification persisted; WAL bytes = AddNode { serial_id: 99 }.
///             Different serial_id from party 1 → bytes differ → mismatch!
///
/// Exec
/// ----
/// `hawk_main` calls `sync_graph_mutations`.  Party 0 calls `build_mutation_bytes`,
/// which contacts parties 1 and 2.  It receives two different byte payloads for the
/// same `modification_id` and bails with:
///
///   "graph mutation mismatch between parties. modification id: 1"
///
/// The first hawk_main task to exit is expected to carry this error.
///
/// Post-conditions
/// ---------------
/// The mismatch is detected and reported in `exec`.
/// DB state is unchanged from setup: party 0 still has 0 WAL rows.
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::wait_for_hawk_failure,
        wal_builder::WalMutationBuilder,
    },
};

const MOD_ID: i64 = 1;
const MISMATCH_ERR: &str = "graph mutation mismatch between parties";

pub struct Wal110 {
    nodes: Option<CpuNodes>,
}

impl Wal110 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal110 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // Two builders for the same modification_id but different node serial_ids.
        // Different serial_ids produce different bincode bytes, triggering the mismatch.
        let builder_a = WalMutationBuilder::new().add_node(MOD_ID, 1, 1); // serial_id = 1
        let builder_b = WalMutationBuilder::new().add_node(MOD_ID, 99, 1); // serial_id = 99

        // WAL seeding:
        //   party 0 — no entry (modification not yet persisted)
        //   party 1 — bytes_a
        //   party 2 — bytes_b  ← conflicts with party 1
        builder_a.insert_mutations(&nodes.0[1].store.graph).await?;
        builder_b.insert_mutations(&nodes.0[2].store.graph).await?;

        // Modification seeding:
        //   party 0 — persisted = FALSE (will try to roll forward)
        //   party 1 — persisted = TRUE  (has bytes_a)
        //   party 2 — persisted = TRUE  (has bytes_b)
        //
        // builder_a is used for the modifications metadata on all parties; the
        // serial_id in the row (1) is consistent and is not consulted during the
        // byte-level mismatch check.
        builder_a
            .seed_modifications_partial(&nodes.0[0].store.graph, 0, 0)
            .await?;
        builder_a
            .seed_modifications_partial(&nodes.0[1].store.graph, 1, 1)
            .await?;
        builder_a
            .seed_modifications_partial(&nodes.0[2].store.graph, 2, 1)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        nodes
            .apply_assertions(&[
                WalAssertions::new().assert_wal_row_count(0),
                WalAssertions::new().assert_wal_row_count(1),
                WalAssertions::new().assert_wal_row_count(1),
            ])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);

        // Wait for the first hawk_main task to exit with an error.
        let failure = wait_for_hawk_failure(&mut hawk_set, Duration::from_secs(30)).await;

        // Cancel remaining tasks (watcher + any still-running servers).
        stop_and_join!(shutdown, hawk_set);

        // Validate the error message.
        let err = failure?;
        eyre::ensure!(
            err.contains(MISMATCH_ERR),
            "unexpected error — wanted mismatch message, got: {err}"
        );
        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // hawk_main bailed before committing any changes; DB state is unchanged.
        nodes
            .apply_assertions(&[
                WalAssertions::new().assert_wal_row_count(0),
                WalAssertions::new().assert_wal_row_count(1),
                WalAssertions::new().assert_wal_row_count(1),
            ])
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
