/// wal_110 — Modification sync: conflict detection (mutation mismatch).
///
/// Mirrors `test_hawk_sync_mutation_mismatch` from
/// `iris-mpc-upgrade-hawk/tests/e2e_hawk.rs`.
///
/// Setup
/// -----
///   Party 0 – modification NOT persisted; no WAL entry.
///             Will attempt to roll forward and request the bytes from party 1/2.
///   Party 1 – modification persisted; WAL bytes built from builder_a (AddNode only;
///             a single node has no neighbors so no AddEdges op is emitted).
///   Party 2 – modification persisted; WAL bytes built from builder_b (AddNode +
///             AddEdges, because builder_b has 2 nodes and node 1 gets node 2 as a
///             neighbor). The extra op makes the serialised bytes differ from party 1
///             → mismatch!
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

use iris_mpc_utils::aws::{AwsClient, AwsClientConfig};
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
        let nodes = CpuNodes::new_clean(&ctx.configs).await?;

        // builder_a: 1 node → AddNode only (no neighbors → no AddEdges emitted).
        // builder_b: 2 nodes → mod_id=1 gets AddNode + AddEdges (node 1 ↔ node 2).
        // The differing ops produce different bincode bytes for mod_id=1, triggering
        // the mismatch.
        let mut builder_a = WalMutationBuilder::new();
        builder_a.add_nodes(1);

        let mut builder_b = WalMutationBuilder::new();
        builder_b.add_nodes(2);

        // Upload fake iris shares to S3 so sync_modifications can fetch them when
        // it rolls forward party 0's unpersisted 'uniqueness' modification.
        let aws_config = AwsClientConfig::new(
            "dev".to_string(),
            ctx.env.public_key_base_url().to_string(),
            "wf-smpcv2-dev-sns-requests".to_string(),
            String::new(),
            0,
            vec![],
        )
        .await;
        let mut aws_client = AwsClient::new(aws_config);
        aws_client.set_public_keyset().await?;
        builder_a.upload_iris_shares(&aws_client).await?;

        // WAL seeding:
        //   party 0 — no entry (modification not yet persisted)
        //   party 1 — builder_a bytes (AddNode only for mod_id=1)
        //   party 2 — builder_b bytes (AddNode + AddEdges for mod_id=1; also mod_id=2)
        builder_a.insert_mutations(&nodes.0[1].store.graph).await?;
        builder_b.insert_mutations(&nodes.0[2].store.graph).await?;

        // Modification seeding:
        //   party 0 — mod_id=1, persisted=FALSE (will try to roll forward)
        //   party 1 — mod_id=1, persisted=TRUE  (has builder_a bytes)
        //   party 2 — mod_id=1 + mod_id=2, persisted=TRUE (has builder_b bytes)
        //
        // builder_a is used for parties 0 and 1; the serial_id in the modifications
        // row (1) is consistent and is not consulted during the byte-level mismatch
        // check.  builder_b seeds both of its entries for party 2.
        builder_a.set_persisted(MOD_ID, false);
        builder_a
            .seed_modifications(&nodes.0[0].store.graph, 0)
            .await?;

        builder_a.set_persisted(MOD_ID, true);
        builder_a
            .seed_modifications(&nodes.0[1].store.graph, 1)
            .await?;

        builder_b
            .seed_modifications(&nodes.0[2].store.graph, 2)
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
                // builder_b inserted 2 entries (mod_id=1 and mod_id=2).
                WalAssertions::new().assert_wal_row_count(2),
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
                WalAssertions::new().assert_wal_row_count(2),
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
