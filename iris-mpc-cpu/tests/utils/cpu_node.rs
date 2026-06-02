use super::CpuConfigs;

// TODO: replace with real imports once open questions #3 and #4 are resolved.
// use iris_mpc_cpu::hnsw::graph::graph_store::{GraphPg, GraphCheckpointRow};
// use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;

/// Per-party database handles used for test setup and post-condition assertions.
///
/// Only the graph store is needed here — WAL mutations and checkpoints both live in
/// the graph store's tables.  There is no iris store because WAL mutations are seeded
/// synthetically (no real iris request pipeline is exercised).
pub struct DbStores {
    // TODO: GraphPg<PlaintextStore>
    // The graph store gives access to:
    //   - hawk_graph_mutations  (WAL table)
    //   - hawk_graph_checkpoints
    //   - persistent state table
}

/// Handles for a single MPC party.
pub struct CpuNode {
    pub stores: DbStores,
}

impl CpuNode {
    pub async fn new(_config: &super::CpuNodeConfig) -> eyre::Result<Self> {
        // TODO:
        //   1. PostgresClient::new(&config.db_url)
        //   2. GraphPg::new(&postgres_client) — runs migrations
        todo!("connect to PostgreSQL and initialize GraphPg")
    }
}

/// All three parties' nodes.  Mirrors `MpcNodes` from the genesis tests.
pub struct CpuNodes(pub [CpuNode; 3]);

impl CpuNodes {
    pub async fn new(configs: &CpuConfigs) -> eyre::Result<Self> {
        // TODO: construct all 3 concurrently (tokio::join! or join_all)
        todo!("construct 3 CpuNode instances from configs")
    }

    /// Run `WalAssertions` against each party.
    pub async fn apply_assertions(
        &self,
        assertions: &[WalAssertions; 3],
    ) -> eyre::Result<()> {
        for (node, assertion) in self.0.iter().zip(assertions.iter()) {
            assertion.assert(&node.stores).await?;
        }
        Ok(())
    }

    /// Assert that every party has exactly `expected` rows in `hawk_graph_checkpoints`.
    pub async fn assert_checkpoint_count(&self, _expected: usize) -> eyre::Result<()> {
        // TODO: GraphPg::recent_checkpoints(large_window) and check len
        todo!("assert checkpoint row count for all parties")
    }

    /// Assert that all 3 parties' latest checkpoint BLAKE3 hashes agree.
    pub async fn assert_checkpoint_hashes_agree(&self) -> eyre::Result<()> {
        // TODO: fetch latest checkpoint row per party, compare blake3_hash fields
        todo!("compare blake3 hashes across parties")
    }

    /// Delete all checkpoint S3 objects and DB rows.  Called in teardown.
    pub async fn cleanup_s3_checkpoints(&self, _configs: &CpuConfigs) -> eyre::Result<()> {
        // TODO: for each party:
        //   1. fetch all checkpoint rows via GraphPg::recent_checkpoints
        //   2. delete S3 object at row.s3_key
        //   3. delete DB row
        todo!("clean up S3 checkpoints for all parties")
    }

    /// Truncate `hawk_graph_mutations` for all parties.  Called in setup/teardown.
    pub async fn truncate_wal(&self) -> eyre::Result<()> {
        // TODO: execute "DELETE FROM {schema}.hawk_graph_mutations" for each party
        todo!("truncate WAL table for all parties")
    }
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// Post-condition assertions for a single party's WAL and checkpoint state.
///
/// All fields are optional — only set fields are checked.  Build with the
/// builder methods or struct literal syntax.
#[derive(Debug, Default)]
pub struct WalAssertions {
    /// Expected number of rows in `hawk_graph_mutations`.
    pub wal_row_count: Option<usize>,
    /// Expected value of `MAX(modification_id)` in `hawk_graph_mutations`.
    pub max_modification_id: Option<i64>,
    /// Expected number of rows in `hawk_graph_checkpoints`.
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
    pub async fn assert(&self, _stores: &DbStores) -> eyre::Result<()> {
        // TODO: implement each field check against the actual DB state
        todo!("run WalAssertions against DbStores")
    }
}
