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
///   1. Detect the mismatch (party 0 max_modification_id = 30; others = 20).
///   2. Transfer mutations 21..=30 from party 0 to parties 1 and 2.
///   3. Allow all three parties to signal ready.
///
/// A subsequent sidecar cycle then materialises each party's WAL independently
/// and reaches 3-party BLAKE3 consensus.  If modification sync resulted in
/// identical in-memory graphs, the sidecar's DB-materialised checkpoints must
/// agree.  A hash mismatch here is a definitive signal that the sync protocol
/// left parties in an inconsistent state.
///
/// Protocol:
///   Setup: seed WAL mods 1..=20 for all parties; seed WAL mods 21..=30 for party 0 only.
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
    },
};

/// Number of node+edge WAL mutations shared by all three parties.
const SHARED_NODES: usize = 10;
const SHARED_EDGES_START: i64 = SHARED_NODES as i64 + 1; // 11
/// Party 0's extra nodes start immediately after the shared edges (21..=25).
const PARTY0_EXTRA_NODES: usize = 5;
const PARTY0_NODES_START: i64 = SHARED_EDGES_START + SHARED_NODES as i64; // 21
/// Party 0's extra edges start immediately after its extra nodes (26..=30).
const PARTY0_EDGES_START: i64 = PARTY0_NODES_START + PARTY0_EXTRA_NODES as i64; // 26

const SHARED_COUNT: usize = SHARED_NODES + SHARED_NODES; // nodes + edges (10 + 10)
const PARTY0_EXTRA: usize = PARTY0_EXTRA_NODES + PARTY0_EXTRA_NODES; // nodes + edges (5 + 5)

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

        // Seed WAL mutations 1..=10 into all three parties.
        let shared_builder = WalMutationBuilder::new()
            .add_nodes_sequential_from(1, SHARED_NODES)
            .add_edges_wrapping(SHARED_NODES, SHARED_EDGES_START);

        shared_builder.build(&nodes).await?;

        // Seed WAL mutations 21..=25 (extra nodes) into party 0 ONLY, simulating a party
        // that committed additional work before the others diverged.
        // IDs start at PARTY0_NODES_START (21) to avoid colliding with shared edge
        // modification IDs 11..=20.
        let party0_builder = (0..PARTY0_EXTRA_NODES as i64)
            .fold(WalMutationBuilder::new(), |b, idx| {
                b.add_node(PARTY0_NODES_START + idx, SHARED_NODES as u32 + idx as u32)
            });

        // Add edges for party 0's extra batch: each node connects to the next two neighbors (wrapping).
        let party0_builder = (0..PARTY0_EXTRA_NODES as i64).fold(party0_builder, |b, idx| {
            let base = idx as u32;
            let num_nodes = PARTY0_EXTRA_NODES as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(PARTY0_EDGES_START + idx, base, vec![neighbor1, neighbor2])
        });

        party0_builder
            .insert_mutations(&nodes.0[0].store.graph)
            .await?;
        party0_builder
            .seed_modifications(&nodes.0[0].store.graph, 0)
            .await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Party 0 has all entries (shared nodes + edges + extra nodes + edges);
        // parties 1 and 2 have only shared nodes + edges.
        let p0_pre = WalAssertions::new()
            .assert_wal_row_count(SHARED_COUNT + PARTY0_EXTRA)
            .assert_max_modification_id(PARTY0_EDGES_START + PARTY0_EXTRA_NODES as i64 - 1)
            .assert_checkpoint_count(0);
        let p12_pre = WalAssertions::new()
            .assert_wal_row_count(SHARED_COUNT)
            .assert_max_modification_id(SHARED_EDGES_START + SHARED_NODES as i64 - 1)
            .assert_checkpoint_count(0);
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

        // One checkpoint per party (sidecar cycle).
        // Max modification_id is based on the last edge added for party 0's extra mutations.
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(PARTY0_EDGES_START + PARTY0_EXTRA_NODES as i64 - 1);
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
