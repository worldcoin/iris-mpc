/// wal_106 — Sidecar reaches consensus even when one party's checkpoint table is behind the others'.
use std::time::Duration;

use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use super::{expect_sidecar_success, run_sidecar};
use crate::utils::{
    cpu_node::{CpuNodes, WalAssertions},
    runner::{CpuTestContext, TestRun},
    wal_builder::WalMutationBuilder,
    MIN_MUTATIONS_PER_SIDECAR_CYCLE,
};

#[derive(Default)]
pub struct Wal106 {
    nodes: Option<CpuNodes>,
    builder: Option<WalMutationBuilder>,
}

impl Wal106 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal106 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs, ctx.s3_client.clone()).await?;

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
            let sidecar_set = run_sidecar(&ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;
        }

        // Introduce desync: delete party 0's newest checkpoint so it lags behind parties 1 and 2.
        nodes.0[0].delete_latest_checkpoint().await?;

        {
            let counts = nodes.checkpoint_counts().await?;
            eyre::ensure!(
                counts == [1, 2, 2],
                "unexpected checkpoint counts after desync: expected [1, 2, 2], got {counts:?}"
            );
        }

        builder.add_nodes(2 * MIN_MUTATIONS_PER_SIDECAR_CYCLE);
        builder.build(nodes).await?;

        // Phase 2: sidecar runs. Wait until each party's checkpoint count exceeds its individual baseline.
        {
            let baselines = nodes.checkpoint_counts().await?;

            let shutdown = CancellationToken::new();
            let sidecar_set = run_sidecar(&ctx.configs, shutdown.clone(), ctx);
            expect_sidecar_success(shutdown, sidecar_set).await?;

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
