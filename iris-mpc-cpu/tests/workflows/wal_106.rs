/// wal_106 — Checkpoint desync: sidecar recovers when one party's checkpoint
/// table is behind the others'.
///
/// After a first sidecar cycle creates a checkpoint at mod_id=10 for all three
/// parties, party 0's newest checkpoint row is manually deleted so that its
/// `genesis_graph_checkpoint` only holds the seeded row (mod_id=0).  Ten more
/// WAL mutations (11..=20) are then seeded for every party.
///
/// A second sidecar cycle must still reach 3-party BLAKE3 consensus and produce
/// a new checkpoint row for every party — even though party 0's "latest
/// checkpoint" (mod_id=0) differs from parties 1 and 2 (mod_id=10).
///
/// Protocol:
///   Phase 1: `sidecar_main` → checkpoint at mod_id=10 (baseline=1).
///   Desync: delete party 0's mod_id=10 checkpoint row.
///   Seed WAL 11..=20 for all parties.
///   Phase 2: `sidecar_main` → checkpoint at mod_id=20 (inline per-party poll).
use std::time::Duration;

use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use super::expect_sidecar_success;
use crate::{
    run_sidecar,
    utils::{
        cpu_node::{CpuNodes, WalAssertions},
        runner::{CpuTestContext, TestRun},
        wal_builder::WalMutationBuilder,
    },
};

/// genesis checkpoint anchor.
const SEED_MOD_ID: i64 = 0;
/// WAL batch consumed by phase-1 sidecar.
const FIRST_BATCH_UP_TO: i64 = 10;
/// WAL batch seeded before phase-2 sidecar.
const SECOND_BATCH_UP_TO: i64 = 20;

const FIRST_BATCH_NODES: usize = (FIRST_BATCH_UP_TO - SEED_MOD_ID) as usize; // 10
const FIRST_BATCH_EDGES_START: i64 = FIRST_BATCH_UP_TO + 1; // 11
const SECOND_BATCH_NODES: usize = (SECOND_BATCH_UP_TO - FIRST_BATCH_UP_TO) as usize; // 10
const SECOND_BATCH_EDGES_START: i64 = SECOND_BATCH_UP_TO + FIRST_BATCH_NODES as i64 + 1; // 31

const FIRST_BATCH_SIZE: usize = FIRST_BATCH_NODES + FIRST_BATCH_NODES; // nodes + edges (10 + 10)
const SECOND_BATCH_SIZE: usize = SECOND_BATCH_NODES + SECOND_BATCH_NODES; // nodes + edges (10 + 10)
const TOTAL_WAL: usize = FIRST_BATCH_SIZE + SECOND_BATCH_SIZE; // 40

pub struct Wal106 {
    nodes: Option<CpuNodes>,
}

impl Wal106 {
    pub fn new() -> Self {
        Self { nodes: None }
    }
}

impl TestRun for Wal106 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // Seed WAL mutations 1..=10.
        let builder = WalMutationBuilder::new()
            .add_nodes_sequential_from(1, FIRST_BATCH_NODES)
            .add_edges_wrapping(FIRST_BATCH_NODES, FIRST_BATCH_EDGES_START);

        builder.build(&nodes).await?;

        // Build checkpoint from WAL up to mod_id = 0 (no mutations at or before 0 →
        // empty graph, matching the genesis state before any mutations were applied).
        nodes.make_checkpoints(SEED_MOD_ID, SEED_MOD_ID).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(FIRST_BATCH_SIZE)
            .assert_max_modification_id(FIRST_BATCH_EDGES_START + FIRST_BATCH_NODES as i64 - 1)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(SEED_MOD_ID);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Phase 1: sidecar checkpoints WAL 1..=10 → checkpoint at mod_id=10.
        // All three parties now have 2 rows: seeded (mod_id=0) + new (mod_id=10).
        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        // Introduce a checkpoint desync: delete party 0's most recent checkpoint row
        // (mod_id=10).  Party 0 is left with only the seeded row (mod_id=0) while
        // parties 1 and 2 retain both rows.
        nodes.0[0].store.delete_latest_checkpoint().await?;

        // Sanity-check: party 0 now has 1 checkpoint; parties 1 and 2 still have 2.
        {
            let counts = nodes.checkpoint_counts().await?;
            eyre::ensure!(
                counts == [1, 2, 2],
                "unexpected checkpoint counts after desync: expected [1, 2, 2], got {counts:?}"
            );
        }

        // Seed additional WAL mutations 11..=20 for all parties.
        // Second batch: serial IDs continue from the first batch, and edges wrap
        // within the full combined graph — too complex for the simple helpers.
        let builder = (FIRST_BATCH_UP_TO + 1..=SECOND_BATCH_UP_TO)
            .fold(WalMutationBuilder::new(), |b, id| {
                b.add_node(id, (id - 1) as u32)
            });
        // Add edges for second batch: each node connects to the next two neighbors
        // (wrapping within the full combined graph, not just this batch).
        let builder = (0..SECOND_BATCH_NODES as i64).fold(builder, |b, idx| {
            let base = (FIRST_BATCH_NODES as u32) + (idx as u32);
            let num_nodes = (FIRST_BATCH_NODES + SECOND_BATCH_NODES) as u32;
            let neighbor1 = (base + 1) % num_nodes;
            let neighbor2 = (base + 2) % num_nodes;
            b.add_edges(
                SECOND_BATCH_EDGES_START + idx,
                base,
                vec![neighbor1, neighbor2],
            )
        });

        builder.build(nodes).await?;

        // Phase 2: sidecar runs again.  Despite the desync, every party must reach
        // 3-party BLAKE3 consensus and insert a new checkpoint row anchored at mod_id=20.
        //
        // Per-party baselines before this run: [1, 2, 2].
        // We wait until each party's count strictly exceeds its individual baseline.
        {
            let baselines = nodes.checkpoint_counts().await?; // [1, 2, 2]

            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);

            let wait_res = timeout(Duration::from_secs(120), async {
                loop {
                    let counts = nodes.checkpoint_counts().await?;
                    let all_advanced = counts
                        .iter()
                        .zip(baselines.iter())
                        .all(|(current, base)| current > base);
                    if all_advanced {
                        break;
                    }
                    sleep(Duration::from_millis(500)).await;
                }
                Ok::<(), eyre::Error>(())
            })
            .await
            .map_err(|_| {
                eyre::eyre!(
                    "timeout: sidecar did not produce new checkpoints for all parties \
                     within 120 s (baselines: {baselines:?})"
                )
            });

            expect_sidecar_success(shutdown, sidecar_set).await?;
            wait_res??;
        }

        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();

        // Party 0: seeded (mod_id=0) + phase-2 new = 2 rows.
        // Parties 1&2: seeded + phase-1 + phase-2 = 3 rows.
        let max_mod_id = SECOND_BATCH_EDGES_START + SECOND_BATCH_NODES as i64 - 1;
        let p0_post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_WAL)
            .assert_max_modification_id(max_mod_id)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(max_mod_id);
        let p12_post = WalAssertions::new()
            .assert_wal_row_count(TOTAL_WAL)
            .assert_max_modification_id(max_mod_id)
            .assert_checkpoint_count(3)
            .assert_latest_checkpoint_mod_id(max_mod_id);
        nodes.apply_split_assertions(&p0_post, &p12_post).await?;

        // All parties must agree on the BLAKE3 hash of the phase-2 checkpoint and
        // verify against the reference hash computed from the full WAL (1..=20).
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
