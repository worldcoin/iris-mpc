/// wal_110 — hawk_main detects and reports incompatible WAL bytes for the same modification_id.
///
/// Parties 1 and 2 have different serialised bytes for mod_id=1 (different graph ops),
/// so party 0's sync request receives mismatched payloads and hawk_main errors out.
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

#[derive(Default)]
pub struct Wal110 {
    nodes: Option<CpuNodes>,
}

impl Wal110 {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TestRun for Wal110 {
    async fn setup(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;
        let aws_client = ctx.make_aws_client().await?;

        // builder_a (party 1): AddNode only for node 1 — no prior nodes, so no AddEdges.
        // builder_b (party 2): AddNode + AddEdges for node 1 via add_node_with_neighbors,
        //   which forces an AddEdges op with a fake neighbor even though no prior nodes
        //   exist.  The two serialised byte payloads for mod_id=1 therefore differ, which
        //   is exactly the incompatibility hawk_main must detect and report.
        let mut builder_a = WalMutationBuilder::new();
        builder_a.add_nodes(1);

        let mut builder_b = WalMutationBuilder::new();
        builder_b.add_node_with_neighbors(&[99]);

        // Insert the differing WAL entries into the respective parties.
        builder_a.insert_mutations(&nodes.0[1].store.graph).await?;
        builder_b.insert_mutations(&nodes.0[2].store.graph).await?;

        // Party 0: modification present but unpersisted — hawk_main must fetch its bytes
        // from the other two parties, where it will find a mismatch.
        builder_a.set_persisted(MOD_ID, false);
        builder_a
            .seed_modifications(&nodes.0[0].store.graph, 0)
            .await?;

        // Party 1: modification persisted (WAL entry from builder_a).
        builder_a.set_persisted(MOD_ID, true);
        builder_a
            .seed_modifications(&nodes.0[1].store.graph, 1)
            .await?;

        // Party 2: modification persisted (WAL entry from builder_b — incompatible bytes).
        builder_b
            .seed_modifications(&nodes.0[2].store.graph, 2)
            .await?;

        // Upload iris shares so hawk_main can locate each modification's payload.
        nodes.init_iris_shares(1, &aws_client).await?;

        self.nodes = Some(nodes);
        Ok(())
    }

    async fn setup_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        nodes
            .apply_assertions(&[
                WalAssertions::new().assert_wal_row_count(0),
                WalAssertions::new().assert_wal_row_count(1),
                // builder_b inserted 1 entry (mod_id=1, AddNode + AddEdges).
                WalAssertions::new().assert_wal_row_count(1),
            ])
            .await
    }

    async fn exec(&mut self, ctx: &CpuTestContext) -> eyre::Result<()> {
        let shutdown = CancellationToken::new();
        let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);

        let failure = wait_for_hawk_failure(&mut hawk_set, Duration::from_secs(30)).await;
        stop_and_join!(shutdown, hawk_set);

        let err = failure?;
        eyre::ensure!(
            err.contains(MISMATCH_ERR),
            "unexpected error — wanted mismatch message, got: {err}"
        );
        Ok(())
    }

    async fn exec_assert(&mut self, _ctx: &CpuTestContext) -> eyre::Result<()> {
        let nodes = self.nodes.as_ref().unwrap();
        // hawk_main exited before writing anything; DB state is unchanged from setup.
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
