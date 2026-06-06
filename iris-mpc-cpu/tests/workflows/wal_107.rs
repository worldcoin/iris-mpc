/// wal_107 — Nontrivial modification sync: hawk_main reconciles a WAL mismatch
/// between parties on startup.
///
/// Party 0 is given 5 extra node+edge mutations beyond what parties 1 and 2
/// have, simulating a scenario where one party processed additional work before
/// an unclean shutdown while the others did not.
///
/// WAL modification ID layout:
///   1–10:   shared node mutations (all parties)
///   11–20:  shared edge mutations (all parties)
///   21–25:  extra node mutations (party 0 only)
///   26–30:  extra edge mutations (party 0 only)
///
/// hawk_main's modification sync protocol must:
///   1. Detect the mismatch
///   2. Transfer mutations
///   3. Allow all three parties to signal ready.
///
/// A subsequent sidecar cycle then materialises each party's WAL independently
/// and reaches 3-party BLAKE3 consensus.  If modification sync resulted in
/// identical in-memory graphs, the sidecar's DB-materialised checkpoints must
/// agree.  A hash mismatch here is a definitive signal that the sync protocol
/// left parties in an inconsistent state.
///
/// Protocol:
///   Setup: seed WAL mods 1..=10 for all parties; seed WAL mods 11..=20 for party 0 only.
///   Phase 1: `hawk_main` → modification sync → signals ready.
///   Phase 2: `sidecar_main` → checkpoint.
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

pub struct Wal107 {
    nodes: Option<CpuNodes>,
}

impl Wal107 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal107 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // seed the wal
        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;
        nodes.make_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        // make one node ahead
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build_single(&nodes.0[0], true, true).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Party 0 has all entries (10 shared nodes + 10 shared edges + 5 extra nodes + 5 extra edges = 30);
        // parties 1 and 2 have only shared nodes + edges (10 + 10 = 20).
        let p0_pre = WalAssertions::new()
            .assert_wal_row_count(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_max_modification_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
            .assert_checkpoint_count(1);
        let p12_pre = WalAssertions::new()
            .assert_wal_row_count(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE)
            .assert_max_modification_id(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64)
            .assert_checkpoint_count(1);
        nodes.apply_split_assertions(&p0_pre, &p12_pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        // Phase 1: hawk_main starts.  The modification sync protocol detects that
        // party 0 is 5 mutations ahead, transfers them to parties 1 and 2, and all
        // three parties signal ready once their graphs converge.
        {
            let shutdown = CancellationToken::new();
            let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
            let res =
                wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await;
            stop_and_join!(shutdown, hawk_set);
            res?;
        }

        // Phase 2: sidecar materialises each party's WAL from the DB and reaches
        // 3-party BLAKE3 consensus.  If modification sync wrote the transferred
        // mutations to DB, all parties' WAL sets are identical and the checkpoint
        // hashes agree.  If sync was in-memory only, party 0's WAL differs from
        // parties 1 and 2 and the consensus round fails — surfacing the bug.
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
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(3 * MIN_MUTATIONS_PER_SIDECAR_CYCLE as i64);
        nodes.apply_uniform_assertions(&post).await?;

        // All three parties must agree on the checkpoint BLAKE3 hash.
        // This is the key assertion: agreement proves modification sync
        // produced identical graphs across parties. Cross-check against the
        // reference materialised from party 0's WAL.
        nodes.assert_consensus_and_reference().await
    }

    async fn teardown(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        if let Some(nodes) = &self.nodes {
            nodes.truncate_checkpoint_tables().await?;
            nodes.cleanup_s3_checkpoints(&ctx.configs).await?;
        }
        Ok(())
    }
}
