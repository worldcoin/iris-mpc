/// wal_109 — Modification-driven sync: roll-forward from staggered per-party state,
///           followed by a sidecar checkpoint cycle.
///
/// Mirrors `test_hawk_init` from `iris-mpc-upgrade-hawk/tests/e2e_hawk.rs`.
///
/// Setup
/// -----
/// All 3 parties hold the same 10 `modifications` rows (mod_ids 1–10).
/// The `persisted` flag and the number of pre-seeded WAL rows differ per party,
/// reflecting the state just after an interrupted insertion pipeline:
///
/// | Party | WAL rows | persisted mods |
/// |-------|----------|----------------|
/// |   0   |     0    |       0        |
/// |   1   |     5    |       5        |
/// |   2   |    10    |      10        |
///
/// Exec — Phase 1
/// ----------------------
/// `hawk_main` calls `sync_graph_mutations`, which transfers the missing mutation
/// bytes from parties that already have them to the parties that do not:
///
/// - Party 0 obtains all 10 mutations (5 from party 1 / party 2, 5 from party 2).
/// - Party 1 obtains mutations 6–10 from party 2.
/// - Party 2 already has all 10 — no change.
///
/// All 3 parties signal ready once their in-memory graphs converge.
///
/// Exec — Phase 2
/// ----------------------
/// `sidecar_main` checkpoints the fully-synced WAL state, writing one checkpoint
/// row per party (anchored at `mod_id = TOTAL_MODS`) and uploading the
/// corresponding S3 object.
///
/// Post-conditions
/// ---------------
/// Every party has 10 rows in `hawk_graph_mutations`, 1 checkpoint row anchored
/// at mod_id 10, a matching S3 object, and all parties agree on the BLAKE3 hash.
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
    },
};

/// Total number of modifications / AddNode WAL entries.
const TOTAL_MODS: i64 = 10;

pub struct Wal109 {
    nodes: Option<CpuNodes>,
}

impl Wal109 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal109 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // One AddNode mutation per modification_id, serial_id = mod_id - 1 (0-indexed).
        let builder10 = (1i64..=TOTAL_MODS).fold(WalMutationBuilder::new(), |b, id| {
            b.add_node(id, (id - 1) as u32)
        });

        // Builder for the first 5 mutations only — seeded into party 1.
        let builder5 = (1i64..=5).fold(WalMutationBuilder::new(), |b, id| {
            b.add_node(id, (id - 1) as u32)
        });

        // WAL seeding:
        //   party 0 — 0 entries (starts with empty WAL; sync will fill it in)
        //   party 1 — 5 entries (mods 1–5)
        //   party 2 — 10 entries (mods 1–10)
        builder5.insert_mutations(&nodes.0[1].store.graph).await?;
        builder10.insert_mutations(&nodes.0[2].store.graph).await?;

        // Modification seeding — ALL parties receive ALL 10 modification rows,
        // but with staggered `persisted` counts mirroring the genesis test:
        //   party 0 → 0 persisted  (all FALSE)
        //   party 1 → 5 persisted  (mods 1–5 TRUE, 6–10 FALSE)
        //   party 2 → 10 persisted (all TRUE)
        for (party_idx, node) in nodes.0.iter().enumerate() {
            builder10
                .seed_modifications_partial(
                    &node.store.graph,
                    party_idx,
                    party_idx * 5, // 0, 5, 10
                )
                .await?;
        }

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        nodes
            .apply_assertions(&[
                WalAssertions::new()
                    .assert_wal_row_count(0)
                    .assert_checkpoint_count(0),
                WalAssertions::new()
                    .assert_wal_row_count(5)
                    .assert_max_modification_id(5)
                    .assert_checkpoint_count(0),
                WalAssertions::new()
                    .assert_wal_row_count(10)
                    .assert_max_modification_id(TOTAL_MODS)
                    .assert_checkpoint_count(0),
            ])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main rolls forward the WAL, syncing all parties to 10
        // mutations.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar_main checkpoints the fully-synced WAL state.
        // baseline = 0 because no checkpoint exists before this phase.
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // All parties hold all 10 mutation rows, 1 checkpoint anchored at the
        // last modification, and a corresponding S3 object (verified automatically).
        let post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_MODS as usize)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(TOTAL_MODS);
        nodes.apply_uniform_assertions(&post).await?;

        // All 3 parties must agree on the BLAKE3 hash of the checkpoint.
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
