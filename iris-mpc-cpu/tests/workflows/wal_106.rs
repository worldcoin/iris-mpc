/// wal_106 — Checkpoint desync: sidecar recovers when one party's checkpoint
/// table is behind the others'.
///
/// After a first sidecar cycle creates a checkpoint for all three parties,
/// party 0's newest checkpoint row is manually deleted so that its
/// `genesis_graph_checkpoint` only holds the seeded row (mod_id=0).  Ten more
/// nodes are then seeded for every party.
///
/// A second sidecar cycle must still reach 3-party BLAKE3 consensus and produce
/// a new checkpoint row for every party — even though party 0's "latest
/// checkpoint" (mod_id=0) differs from parties 1 and 2.
///
/// Protocol:
///   Phase 1: `sidecar_main` → checkpoint at end of first batch (baseline=1).
///   Desync: delete party 0's newest checkpoint row.
///   Seed 10 more nodes for all parties.
///   Phase 2: `sidecar_main` → checkpoint at end of second batch (inline per-party poll).
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
        MIN_MUTATIONS_PER_SIDECAR_CYCLE,
    },
};

pub struct Wal106 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal106 {
    pub fn new() -> Self {
        Self {
            nodes: None,
            builder: None,
        }
    }
}

impl TestRun for Wal106 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        let mut builder = WalMutationBuilder::new();
        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        nodes.make_checkpoints().await?;

        builder.add_nodes(MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(&nodes).await?;

        self.nodes = Some(nodes);
        self.builder = Some(builder);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let pre = WalAssertions::new()
            .assert_wal_row_count(10)
            .assert_max_modification_id(10)
            .assert_checkpoint_count(1)
            .assert_latest_checkpoint_mod_id(5);
        nodes.apply_uniform_assertions(&pre).await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        let builder = self.builder.as_mut().unwrap();

        {
            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar!(ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        // Introduce a checkpoint desync: delete party 0's most recent checkpoint row.
        // Party 0 is left with only the seeded row (mod_id=0) while parties 1 and 2
        // retain both rows.
        nodes.0[0].delete_latest_checkpoint().await?;

        // Sanity-check: party 0 now has 1 checkpoint; parties 1 and 2 still have 2.
        {
            let counts = nodes.checkpoint_counts().await?;
            eyre::ensure!(
                counts == [1, 2, 2],
                "unexpected checkpoint counts after desync: expected [1, 2, 2], got {counts:?}"
            );
        }

        // Seed additional WAL mutations for all parties.
        builder.add_nodes(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(nodes).await?;

        // Phase 2: sidecar runs again.  Despite the desync, every party must reach
        // 3-party BLAKE3 consensus and insert a new checkpoint row anchored at mod_id=40.
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

        let p0_post = WalAssertions::new()
            .assert_wal_row_count(20)
            .assert_max_modification_id(20)
            .assert_checkpoint_count(2)
            .assert_latest_checkpoint_mod_id(20);
        let p12_post = WalAssertions::new()
            .assert_wal_row_count(20)
            .assert_max_modification_id(20)
            .assert_checkpoint_count(3)
            .assert_latest_checkpoint_mod_id(20);
        nodes.apply_split_assertions(&p0_post, &p12_post).await?;

        // All parties must agree on the BLAKE3 hash of the phase-2 checkpoint and
        // verify against the reference hash computed from the full WAL.
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
