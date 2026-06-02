use super::CpuConfigs;

use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::graph::graph_store::{GraphCheckpointRow, GraphPg};

/// Per-party database handles for test setup and post-condition assertions.
///
/// Only the graph store is needed — WAL mutations and checkpoints both live in
/// the graph store's tables.  There is no iris store because WAL mutations are seeded
/// synthetically.
///
/// The checkpoint format (GraphV4 / bincode) is store-agnostic, so `PlaintextStore`
/// is safe to use here even though `hawk_main` uses `Aby3Store` at runtime.
pub struct DbStores {
    pub graph: GraphPg<PlaintextStore>,
}

impl DbStores {
    pub async fn new(config: &super::CpuNodeConfig) -> eyre::Result<Self> {
        let pg =
            PostgresClient::new(&config.db_url, &config.db_schema, AccessMode::ReadWrite).await?;
        pg.migrate().await;
        let graph = GraphPg::new(&pg).await?;
        Ok(Self { graph })
    }

    /// Count rows in `genesis_graph_checkpoint`.
    pub async fn count_checkpoints(&self) -> eyre::Result<usize> {
        self.graph
            .get_genesis_graph_checkpoints()
            .await
            .map(|r| r.len())
    }

    /// Get the latest checkpoint row, returning None if none exists.
    pub async fn latest_checkpoint(&self) -> eyre::Result<Option<GraphCheckpointRow>> {
        self.graph.get_latest_genesis_graph_checkpoint().await
    }

    /// Head-check the latest checkpoint's S3 key to confirm the object exists.
    pub async fn verify_latest_checkpoint_s3_object(&self, bucket: &str) -> eyre::Result<()> {
        let row = self
            .latest_checkpoint()
            .await?
            .ok_or_else(|| eyre!("no checkpoint row"))?;
        let s3 = build_s3_client(bucket).await?;
        s3.head_object()
            .bucket(bucket)
            .key(&row.s3_key)
            .send()
            .await
            .map_err(|e| eyre!("S3 object {} missing: {}", row.s3_key, e))?;
        Ok(())
    }

    /// Truncate all WAL and checkpoint tables for this party.
    /// Covers: hawk_graph_mutations, genesis_graph_checkpoint, and the persistent state table.
    pub async fn truncate_checkpoint_tables(&self) -> eyre::Result<()> {
        // TODO (open question #4): confirm persistent state table name.
        // Execute for each table:
        //   sqlx::query("TRUNCATE {schema}.hawk_graph_mutations").execute(&pool).await?;
        //   sqlx::query("TRUNCATE {schema}.genesis_graph_checkpoint").execute(&pool).await?;
        //   sqlx::query("TRUNCATE {schema}.{persistent_state_table}").execute(&pool).await?;
        todo!("truncate hawk_graph_mutations, genesis_graph_checkpoint, persistent state")
    }

    /// Count rows in `hawk_graph_mutations`.
    pub async fn wal_row_count(&self) -> eyre::Result<usize> {
        // TODO: SELECT COUNT(*) FROM {schema}.hawk_graph_mutations
        todo!("count hawk_graph_mutations rows")
    }

    /// Get MAX(modification_id) from `hawk_graph_mutations`, returns None if empty.
    pub async fn max_modification_id(&self) -> eyre::Result<Option<i64>> {
        // TODO: SELECT MAX(modification_id) FROM {schema}.hawk_graph_mutations
        todo!("get max modification_id from hawk_graph_mutations")
    }
}

/// Handles for a single MPC party.
pub struct CpuNode {
    pub stores: DbStores,
}

impl CpuNode {
    pub async fn new(config: &super::CpuNodeConfig) -> eyre::Result<Self> {
        let stores = DbStores::new(config).await?;
        Ok(Self { stores })
    }
}

/// All three parties' nodes.  Mirrors `MpcNodes` from the genesis tests.
pub struct CpuNodes(pub [CpuNode; 3]);

impl CpuNodes {
    pub async fn new(configs: &CpuConfigs) -> eyre::Result<Self> {
        // Construct all 3 concurrently.
        let (n0, n1, n2) = tokio::try_join!(
            CpuNode::new(&configs[0]),
            CpuNode::new(&configs[1]),
            CpuNode::new(&configs[2]),
        )?;
        Ok(Self([n0, n1, n2]))
    }

    /// Run `WalAssertions` against each party in sequence.
    pub async fn apply_assertions(&self, assertions: &[WalAssertions; 3]) -> eyre::Result<()> {
        for (node, assertion) in self.0.iter().zip(assertions.iter()) {
            assertion.assert(&node.stores).await?;
        }
        Ok(())
    }

    /// Assert that every party has exactly `expected` rows in `genesis_graph_checkpoint`.
    pub async fn assert_checkpoint_count(&self, expected: usize) -> eyre::Result<()> {
        for (i, node) in self.0.iter().enumerate() {
            let count = node.stores.count_checkpoints().await?;
            eyre::ensure!(
                count == expected,
                "party {i}: expected {expected} checkpoint rows, got {count}"
            );
        }
        Ok(())
    }

    /// Assert that all 3 parties' latest checkpoint BLAKE3 hashes are equal.
    pub async fn assert_checkpoint_hashes_agree(&self) -> eyre::Result<()> {
        // TODO (open question #3): once GraphCheckpointRow is available, fetch
        // latest row per party and compare blake3_hash fields.
        //
        // let hashes: Vec<String> = ...
        // ensure all are equal
        todo!("compare blake3_hash across all 3 parties' latest checkpoint rows")
    }

    /// Assert that all 3 parties' latest checkpoint BLAKE3 hashes match a
    /// reference hash computed by the test itself.
    ///
    /// See open question #7 in readme for the WAL materialization API needed here.
    pub async fn assert_checkpoint_hashes_match_reference(
        &self,
        _reference_hash: &[u8; 32],
    ) -> eyre::Result<()> {
        // TODO (open question #7): materialize graph from WAL in test process,
        // compute blake3 hash, then compare against each party's stored blake3_hash.
        todo!("compare stored hashes against test-computed reference hash")
    }

    /// Return the checkpoint row count for each party.
    /// Used by wait_for_new_checkpoint to poll for TC-2 progress.
    pub async fn checkpoint_counts(&self) -> eyre::Result<[usize; 3]> {
        let c0 = self.0[0].stores.count_checkpoints().await?;
        let c1 = self.0[1].stores.count_checkpoints().await?;
        let c2 = self.0[2].stores.count_checkpoints().await?;
        Ok([c0, c1, c2])
    }

    /// Delete all checkpoint S3 objects and DB rows.  Called in teardown.
    pub async fn cleanup_s3_checkpoints(&self, _configs: &CpuConfigs) -> eyre::Result<()> {
        // TODO: for each party:
        //   1. fetch all rows via self.graph.recent_checkpoints(usize::MAX)
        //   2. delete each S3 object at row.s3_key
        //   3. delete DB rows (or truncate genesis_graph_checkpoint)
        todo!("delete S3 checkpoint objects and truncate genesis_graph_checkpoint for all parties")
    }

    /// Truncate WAL and checkpoint tables for all parties.
    pub async fn truncate_checkpoint_tables(&self) -> eyre::Result<()> {
        for node in &self.0 {
            node.stores.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// Post-condition assertions for a single party's WAL and checkpoint state.
///
/// All fields are optional — only set fields are checked.
#[derive(Debug, Default, Clone)]
pub struct WalAssertions {
    /// Expected number of rows in `hawk_graph_mutations`.
    pub wal_row_count: Option<usize>,
    /// Expected value of `MAX(modification_id)` in `hawk_graph_mutations`.
    pub max_modification_id: Option<i64>,
    /// Expected number of rows in `genesis_graph_checkpoint`.
    pub checkpoint_count: Option<usize>,
    /// Expected `graph_mutation_id` of the latest checkpoint row.
    pub latest_checkpoint_mod_id: Option<i64>,
    /// If Some(true), verify the latest checkpoint's S3 object exists.
    pub s3_object_exists: Option<bool>,
}

impl WalAssertions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assert_wal_row_count(mut self, n: usize) -> Self {
        self.wal_row_count = Some(n);
        self
    }

    pub fn assert_max_modification_id(mut self, id: i64) -> Self {
        self.max_modification_id = Some(id);
        self
    }

    pub fn assert_checkpoint_count(mut self, n: usize) -> Self {
        self.checkpoint_count = Some(n);
        self
    }

    pub fn assert_latest_checkpoint_mod_id(mut self, id: i64) -> Self {
        self.latest_checkpoint_mod_id = Some(id);
        self
    }

    pub fn assert_s3_object_exists(mut self, exists: bool) -> Self {
        self.s3_object_exists = Some(exists);
        self
    }

    /// Run all set assertions against `stores`.
    pub async fn assert(&self, stores: &DbStores) -> eyre::Result<()> {
        if let Some(expected) = self.wal_row_count {
            let actual = stores.wal_row_count().await?;
            eyre::ensure!(
                actual == expected,
                "WAL row count: expected {expected}, got {actual}"
            );
        }

        if let Some(expected) = self.max_modification_id {
            let actual = stores.max_modification_id().await?;
            eyre::ensure!(
                actual == Some(expected),
                "max modification_id: expected Some({expected}), got {actual:?}"
            );
        }

        if let Some(expected) = self.checkpoint_count {
            let actual = stores.count_checkpoints().await?;
            eyre::ensure!(
                actual == expected,
                "checkpoint count: expected {expected}, got {actual}"
            );
        }

        if let Some(expected_mod_id) = self.latest_checkpoint_mod_id {
            // TODO (open question #3): once GraphCheckpointRow is available:
            //   let row = stores.latest_checkpoint().await?.ok_or_else(|| eyre!("no checkpoint row"))?;
            //   ensure!(row.graph_mutation_id == Some(expected_mod_id), ...);
            let _ = expected_mod_id;
            todo!("assert latest_checkpoint.graph_mutation_id == expected_mod_id")
        }

        if let Some(true) = self.s3_object_exists {
            // TODO: stores.verify_latest_checkpoint_s3_object(bucket).await?
            // (bucket must be threaded through or stored on DbStores)
            todo!("assert S3 object exists for latest checkpoint")
        }

        Ok(())
    }
}
