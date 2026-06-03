/// wal_107 — Nontrivial modification sync: hawk_main reconciles a WAL mismatch
/// between parties on startup.
///
/// Party 0 is given mutations 1..=15 while parties 1 and 2 only have 1..=10,
/// simulating a scenario where one party processed additional work before an
/// unclean shutdown while the others did not.
///
/// hawk_main's modification sync protocol must:
///   1. Detect the mismatch (party 0 max_modification_id = 15; others = 10).
///   2. Transfer mutations 11..=15 from party 0 to parties 1 and 2.
///   3. Allow all three parties to signal ready.
///
/// A subsequent sidecar cycle then materialises each party's WAL independently
/// and reaches 3-party BLAKE3 consensus.  If modification sync resulted in
/// identical in-memory graphs, the sidecar's DB-materialised checkpoints must
/// agree.  A hash mismatch here is a definitive signal that the sync protocol
/// left parties in an inconsistent state.
///
/// Protocol:
///   Setup: seed WAL 1..=10 for all parties; seed WAL 11..=15 for party 0 only.
///   Phase 1: `hawk_main` → modification sync → signals ready.
///   Phase 2: `sidecar_main` → checkpoint.
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use crate::{
    run_hawk, run_sidecar, stop_and_join,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wait_conditions::{wait_for_all_ready, wait_for_new_checkpoint},
        wal_builder::WalMutationBuilder,
    },
};

/// WAL entries present on all three parties before the divergence.
const SHARED_UP_TO_MOD_ID: i64 = 10;
/// Additional WAL entries present only on party 0.
const PARTY0_UP_TO_MOD_ID: i64 = 15;

const SHARED_NODES: usize = SHARED_UP_TO_MOD_ID as usize; // 10
const SHARED_EDGES_START: i64 = SHARED_UP_TO_MOD_ID + 1; // 11
const PARTY0_EXTRA_NODES: usize = (PARTY0_UP_TO_MOD_ID - SHARED_UP_TO_MOD_ID) as usize; // 5
const PARTY0_EDGES_START: i64 = PARTY0_UP_TO_MOD_ID + SHARED_NODES as i64 + 1; // 26

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
        let nodes = CpuNodes::new(&ctx.configs).await?;
        nodes.truncate_checkpoint_tables().await?;

        // Seed WAL mutations 1..=10 into all three parties.
        let shared_builder = (1i64..=SHARED_UP_TO_MOD_ID)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - 1) as u32, 1)
            });

        // Add edges for shared batch: each node connects to the next two neighbors (wrapping).
        let shared_builder = (0..SHARED_NODES as i64).fold(shared_builder, |b, idx| {
            let base = idx as u32;
            let num_nodes = SHARED_NODES as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(
                SHARED_EDGES_START + idx,
                base,
                vec![neighbor1, neighbor2],
                0,
            )
        });

        shared_builder.build(&nodes).await?;

        // Seed WAL mutations 11..=15 into party 0 ONLY, simulating a party that
        // committed additional work before the others diverged.
        let party0_builder = (SHARED_UP_TO_MOD_ID + 1..=PARTY0_UP_TO_MOD_ID)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - 1) as u32, 1)
            });

        // Add edges for party 0's extra batch: each node connects to the next two neighbors (wrapping).
        let party0_builder = (0..PARTY0_EXTRA_NODES as i64).fold(party0_builder, |b, idx| {
            let base = idx as u32;
            let num_nodes = PARTY0_EXTRA_NODES as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(
                PARTY0_EDGES_START + idx,
                base,
                vec![neighbor1, neighbor2],
                0,
            )
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
        nodes
            .apply_assertions(&[p0_pre, p12_pre.clone(), p12_pre])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

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
            let mut sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            let res = wait_for_new_checkpoint(
                nodes,
                &ctx.configs,
                /* baseline */ 0,
                Duration::from_secs(120),
            )
            .await;
            stop_and_join!(shutdown, sidecar_set);
            res?;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // One checkpoint per party (sidecar cycle).  S3 objects must exist.
        // Max modification_id is based on the last edge added for party 0's extra mutations.
        let post = WalAssertions::new()
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(PARTY0_EDGES_START + PARTY0_EXTRA_NODES as i64 - 1)
            .assert_s3_object_exists(true);
        nodes
            .apply_assertions(&[post.clone(), post.clone(), post])
            .await?;

        // All three parties must agree on the checkpoint BLAKE3 hash.
        // This is the key assertion: agreement proves modification sync
        // produced identical graphs across parties.
        nodes.assert_checkpoint_hashes_agree().await?;

        // Cross-check against the reference materialised from party 0's WAL,
        // which is the ground truth (party 0 had the complete mutation set).
        let reference_hash = nodes.0[0].store.compute_reference_hash().await?;
        nodes
            .assert_checkpoint_hashes_match_reference(&reference_hash)
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
