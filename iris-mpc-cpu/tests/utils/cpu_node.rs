use crate::utils::CpuNodeConfig;

use super::CpuConfigs;

use eyre::eyre;
use iris_mpc_common::{
    postgres::{AccessMode, PostgresClient},
    IrisVectorId,
};
use iris_mpc_cpu::graph_checkpoint::stream_serialize_and_upload_with;
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::graph::graph_store::{GraphCheckpointRow, GraphPg};
use iris_mpc_cpu::hnsw::GraphMem;

/// Per-party database handles for test setup and post-condition assertions.
///
/// Only the graph store is needed — WAL mutations and checkpoints both live in
/// the graph store's tables.  There is no iris store because WAL mutations are seeded
/// synthetically.
///
/// The checkpoint format (`bincode([GraphMem; 2])`) is store-agnostic, so `PlaintextStore`
/// is safe to use here even though `hawk_main` uses `Aby3Store` at runtime.
pub struct DbStores {
    pub graph: GraphPg<PlaintextStore>,
}

impl DbStores {
    pub async fn new(config: &CpuNodeConfig) -> eyre::Result<Self> {
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

    /// Truncate all WAL and checkpoint tables for this party.
    /// Covers: hawk_graph_mutations, genesis_graph_checkpoint, persistent_state, and modifications.
    pub async fn truncate_checkpoint_tables(&self) -> eyre::Result<()> {
        sqlx::query("TRUNCATE hawk_graph_mutations")
            .execute(self.graph.pool())
            .await?;
        sqlx::query("TRUNCATE genesis_graph_checkpoint")
            .execute(self.graph.pool())
            .await?;
        sqlx::query("TRUNCATE persistent_state")
            .execute(self.graph.pool())
            .await?;
        sqlx::query("TRUNCATE modifications")
            .execute(self.graph.pool())
            .await?;
        Ok(())
    }

    /// Count rows in `hawk_graph_mutations`.
    pub async fn wal_row_count(&self) -> eyre::Result<usize> {
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM hawk_graph_mutations")
            .fetch_one(self.graph.pool())
            .await?;
        Ok(count as usize)
    }

    /// Delete the single most-recently inserted `genesis_graph_checkpoint` row.
    ///
    /// Uses `MAX(id)` as the tiebreaker so it always removes exactly the row
    /// that `latest_checkpoint()` would return.  A no-op when the table is empty.
    ///
    /// Intended for checkpoint-desync tests that need to artificially regress one
    /// party's checkpoint state.
    pub async fn delete_latest_checkpoint(&self) -> eyre::Result<()> {
        sqlx::query(
            "DELETE FROM genesis_graph_checkpoint \
             WHERE id = (SELECT MAX(id) FROM genesis_graph_checkpoint)",
        )
        .execute(self.graph.pool())
        .await?;
        Ok(())
    }

    /// Get MAX(modification_id) from `hawk_graph_mutations`, returns None if empty.
    pub async fn max_modification_id(&self) -> eyre::Result<Option<i64>> {
        let max: Option<i64> =
            sqlx::query_scalar("SELECT MAX(modification_id) FROM hawk_graph_mutations")
                .fetch_one(self.graph.pool())
                .await?;
        Ok(max)
    }

    /// Materialise all WAL rows into a fresh `[GraphMem; 2]` and return the BLAKE3
    /// digest of its bincode serialisation.
    ///
    /// This mirrors exactly what the sidecar does in `terminal.rs`:
    /// `bincode::serialize_into(writer, &graph)` where `graph: BothEyes<GraphMem<VectorId>>`.
    /// Use this to build a reference hash for `CpuNodes::assert_checkpoint_hashes_match_reference`.
    pub async fn compute_reference_hash(&self) -> eyre::Result<[u8; 32]> {
        let rows = self.graph.get_hawk_graph_mutations_after(None).await?;
        let mut graph_pair: [GraphMem<IrisVectorId>; 2] = [GraphMem::new(), GraphMem::new()];
        for row in &rows {
            let [left_muts, right_muts] = row.deserialize_mutations()?;
            graph_pair[0].insert_apply_all(&left_muts)?;
            graph_pair[1].insert_apply_all(&right_muts)?;
        }
        let bytes = bincode::serialize(&graph_pair)?;
        Ok(*blake3::hash(&bytes).as_bytes())
    }
}

/// Handles for a single MPC party.
pub struct CpuNode {
    pub store: DbStores,
    pub config: CpuNodeConfig,
    pub s3: aws_sdk_s3::Client,
}

impl CpuNode {
    pub async fn new(config: CpuNodeConfig) -> eyre::Result<Self> {
        let stores = DbStores::new(&config).await?;
        let aws_config = aws_config::load_from_env().await;
        let s3 = aws_sdk_s3::Client::new(&aws_config);
        Ok(Self {
            store: stores,
            s3,
            config,
        })
    }

    /// Verify that all checkpoint S3 objects exist.
    ///
    /// Reads all checkpoints from the store and head-checks each one's S3 key
    /// to confirm the object exists in the bucket.
    pub async fn verify_all_checkpoints_in_s3(&self, bucket: &str) -> eyre::Result<()> {
        let checkpoints = self.store.graph.get_genesis_graph_checkpoints().await?;
        for row in checkpoints {
            self.s3
                .head_object()
                .bucket(bucket)
                .key(&row.s3_key)
                .send()
                .await
                .map_err(|e| eyre!("S3 object {} missing: {}", row.s3_key, e))?;
        }
        Ok(())
    }

    /// List all S3 object keys in the given bucket (all pages, no prefix filter).
    ///
    /// Returns an empty vec when the bucket is empty.  Intended for exec_assert
    /// checks that verify the pruning pass reduced the number of stored checkpoints.
    pub async fn list_s3_keys(&self, bucket: &str) -> eyre::Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;
        loop {
            let mut req = self.s3.list_objects_v2().bucket(bucket);
            if let Some(ref token) = continuation_token {
                req = req.continuation_token(token);
            }
            let resp = req
                .send()
                .await
                .map_err(|e| eyre!("failed to list S3 objects in bucket {bucket}: {e}"))?;
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    keys.push(key.to_string());
                }
            }
            if resp.is_truncated().unwrap_or(false) {
                continuation_token = resp.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }
        Ok(keys)
    }

    /// Seed a genesis (seed) checkpoint for this party.
    pub async fn make_checkpoint(&self) -> eyre::Result<GraphCheckpointRow> {
        self.make_checkpoint_with_options(false).await
    }

    /// Seed an archival checkpoint for this party.
    pub async fn make_archival_checkpoint(&self) -> eyre::Result<GraphCheckpointRow> {
        self.make_checkpoint_with_options(true).await
    }

    /// Build and persist a genesis checkpoint for this party.
    ///
    /// Replays all WAL mutations from the DB to construct the graph state, then
    /// serializes and uploads it to S3. The most recent `modification_id` is
    /// looked up automatically and used as both `last_indexed_iris_id` and
    /// `last_indexed_modification_id`.
    ///
    /// 1. Query `MAX(modification_id)` from `hawk_graph_mutations`.
    /// 2. Read all `hawk_graph_mutations` rows up to that id.
    /// 3. Replay mutations into `[GraphMem<IrisVectorId>; 2]` (left + right eye).
    /// 4. Serialize with `bincode::serialize` — same wire format as the sidecar.
    /// 5. Compute BLAKE3 hash of the bytes.
    /// 6. Upload to S3 at key `"{party_id}/checkpoints/{label}_{last_modification_id}"`,
    ///    where `label` is `"archival"` when `is_archival` is true, otherwise `"seed"`.
    /// 7. Insert `genesis_graph_checkpoint` DB row inside a transaction.
    /// 8. Return the newly inserted `GraphCheckpointRow`.
    async fn make_checkpoint_with_options(
        &self,
        is_archival: bool,
    ) -> eyre::Result<GraphCheckpointRow> {
        let bucket = &self.config.checkpoint_bucket;
        let party_id = self.config.party_id;
        let last_modification_id = self
            .store
            .graph
            .get_max_hawk_graph_mutation_id()
            .await?
            .ok_or_else(|| eyre!("no mutations found; cannot create checkpoint"))?;
        let last_iris_id = last_modification_id;
        // Build the graph by replaying WAL mutations up to last_modification_id.
        // The checkpoint format is a bincode-serialised pair of GraphMem (left + right eye),
        // matching what the sidecar writes in terminal.rs and the materializer reads.
        let mutation_rows = self
            .store
            .graph
            .get_hawk_graph_mutations(Some(last_modification_id))
            .await?;
        let mut left_graph = GraphMem::<IrisVectorId>::new();
        let mut right_graph = GraphMem::<IrisVectorId>::new();
        for row in &mutation_rows {
            let both_eyes = row.deserialize_mutations()?;
            left_graph.insert_apply_all(&both_eyes[0])?;
            right_graph.insert_apply_all(&both_eyes[1])?;
        }
        let graph_pair: [GraphMem<IrisVectorId>; 2] = [left_graph, right_graph];
        let bytes = bincode::serialize(&graph_pair)?;
        let hash_hex = hex::encode(blake3::hash(&bytes).as_bytes());
        let label = if is_archival { "archival" } else { "seed" };
        let s3_key = format!("{party_id}/checkpoints/{label}_{last_modification_id}");

        stream_serialize_and_upload_with(
            &self.s3,
            bucket,
            &s3_key,
            move |w| w.write_all(&bytes).map_err(|e| eyre::eyre!(e)),
            8 * 1024 * 1024,
            4,
        )
        .await?;

        let mut tx = self.store.graph.begin_tx().await?;
        GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
            &mut tx,
            &s3_key,
            last_iris_id,
            last_modification_id,
            Some(last_modification_id),
            &hash_hex,
            is_archival,
            4,
        )
        .await?;
        tx.commit().await?;

        self.store
            .graph
            .get_latest_genesis_graph_checkpoint()
            .await?
            .ok_or_else(|| eyre!("no checkpoint row found after insert"))
    }

    /// Run all set assertions in `assertions` against this node.
    ///
    /// Always verifies that all checkpoint S3 objects exist in the bucket.
    pub async fn assert(&self, assertions: &WalAssertions) -> eyre::Result<()> {
        let store = &self.store;
        if let Some(expected) = assertions.wal_row_count {
            let actual = store.wal_row_count().await?;
            eyre::ensure!(
                actual == expected,
                "WAL row count: expected {expected}, got {actual}"
            );
        }

        if let Some(expected) = assertions.max_modification_id {
            let actual = store.max_modification_id().await?;
            eyre::ensure!(
                actual == Some(expected),
                "max modification_id: expected Some({expected}), got {actual:?}"
            );
        }

        if let Some(expected) = assertions.checkpoint_count {
            let actual = store.count_checkpoints().await?;
            eyre::ensure!(
                actual == expected,
                "checkpoint count: expected {expected}, got {actual}"
            );

            // When checkpoint count is asserted, also verify S3 object count matches.
            let s3_count = self.list_s3_keys(&self.config.checkpoint_bucket).await?.len();
            eyre::ensure!(
                s3_count == expected,
                "S3 object count in bucket {}: expected {expected}, got {s3_count}",
                self.config.checkpoint_bucket
            );
        }

        if let Some(expected_mod_id) = assertions.latest_checkpoint_mod_id {
            let row = store
                .latest_checkpoint()
                .await?
                .ok_or_else(|| eyre!("no checkpoint row found"))?;
            eyre::ensure!(
                row.graph_mutation_id == Some(expected_mod_id),
                "latest checkpoint graph_mutation_id: expected Some({expected_mod_id}), got {:?}",
                row.graph_mutation_id
            );
        }

        // Always verify all checkpoint S3 objects exist.
        self.verify_all_checkpoints_in_s3(&self.config.checkpoint_bucket)
            .await?;

        Ok(())
    }
}

/// All three parties' nodes.  Mirrors `MpcNodes` from the genesis tests.
pub struct CpuNodes(pub [CpuNode; 3]);

impl CpuNodes {
    pub async fn new(configs: &CpuConfigs) -> eyre::Result<Self> {
        // Construct all 3 concurrently.
        let (n0, n1, n2) = tokio::try_join!(
            CpuNode::new(configs[0].clone()),
            CpuNode::new(configs[1].clone()),
            CpuNode::new(configs[2].clone()),
        )?;
        Ok(Self([n0, n1, n2]))
    }

    /// Create nodes and immediately truncate checkpoint tables and clear S3 buckets —
    /// the standard starting state for every workflow test.
    pub async fn new_clean(configs: &CpuConfigs) -> eyre::Result<Self> {
        let nodes = Self::new(configs).await?;
        nodes.clear_all_s3_buckets(configs).await?;
        nodes.truncate_checkpoint_tables().await?;
        Ok(nodes)
    }

    /// Run `WalAssertions` against each party in sequence.
    pub async fn apply_assertions(&self, assertions: &[WalAssertions; 3]) -> eyre::Result<()> {
        for (node, assertion) in self.0.iter().zip(assertions.iter()) {
            node.assert(assertion).await?;
        }
        Ok(())
    }

    /// Assert that every party has exactly `expected` rows in `genesis_graph_checkpoint`.
    pub async fn assert_checkpoint_count(&self, expected: usize) -> eyre::Result<()> {
        for (i, node) in self.0.iter().enumerate() {
            let count = node.store.count_checkpoints().await?;
            eyre::ensure!(
                count == expected,
                "party {i}: expected {expected} checkpoint rows, got {count}"
            );
        }
        Ok(())
    }

    /// Assert that all 3 parties' latest checkpoint BLAKE3 hashes are equal.
    pub async fn assert_checkpoint_hashes_agree(&self) -> eyre::Result<()> {
        let mut hashes: Vec<String> = Vec::with_capacity(3);
        for (i, node) in self.0.iter().enumerate() {
            let row = node
                .store
                .latest_checkpoint()
                .await?
                .ok_or_else(|| eyre!("party {i}: no checkpoint row found"))?;
            hashes.push(row.blake3_hash);
        }
        let first = &hashes[0];
        eyre::ensure!(
            hashes.iter().all(|h| h == first),
            "checkpoint blake3_hash mismatch across parties: {:?}",
            hashes
        );
        Ok(())
    }

    /// Assert that all 3 parties' latest checkpoint BLAKE3 hashes match a
    /// reference hash computed by the test itself.
    ///
    /// Build the reference with `DbStores::compute_reference_hash()` on any
    /// one party (all three have the same WAL content after seeding).
    pub async fn assert_checkpoint_hashes_match_reference(
        &self,
        reference_hash: &[u8; 32],
    ) -> eyre::Result<()> {
        let hash_hex = hex::encode(reference_hash);
        for (i, node) in self.0.iter().enumerate() {
            let row = node
                .store
                .latest_checkpoint()
                .await?
                .ok_or_else(|| eyre!("party {i}: no checkpoint row found"))?;
            eyre::ensure!(
                row.blake3_hash == hash_hex,
                "party {i}: checkpoint hash mismatch: expected {hash_hex}, got {}",
                row.blake3_hash
            );
        }
        Ok(())
    }

    /// Return the checkpoint row count for each party.
    /// Used by wait_for_new_checkpoint to poll checkpoint progress.
    pub async fn checkpoint_counts(&self) -> eyre::Result<[usize; 3]> {
        let c0 = self.0[0].store.count_checkpoints().await?;
        let c1 = self.0[1].store.count_checkpoints().await?;
        let c2 = self.0[2].store.count_checkpoints().await?;
        Ok([c0, c1, c2])
    }

    /// Delete all checkpoint S3 objects and DB rows.  Called in teardown.
    pub async fn cleanup_s3_checkpoints(&self, configs: &CpuConfigs) -> eyre::Result<()> {
        for (node, config) in self.0.iter().zip(configs.iter()) {
            let rows = node.store.graph.get_genesis_graph_checkpoints().await?;
            for row in &rows {
                // Ignore errors — the object may already be gone.
                let _ = node
                    .s3
                    .delete_object()
                    .bucket(&config.checkpoint_bucket)
                    .key(&row.s3_key)
                    .send()
                    .await;
            }
            node.store.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }

    /// Truncate WAL and checkpoint tables for all parties.
    pub async fn truncate_checkpoint_tables(&self) -> eyre::Result<()> {
        for node in &self.0 {
            node.store.truncate_checkpoint_tables().await?;
        }
        Ok(())
    }

    /// Seed a genesis checkpoint for all 3 parties concurrently.
    ///
    /// Each node pulls its own `checkpoint_bucket` and `party_id` from its
    /// internal config — callers do not need to pass a `CpuConfigs`.
    pub async fn make_checkpoints(&self) -> eyre::Result<[GraphCheckpointRow; 3]> {
        let (r0, r1, r2) = tokio::try_join!(
            self.0[0].make_checkpoint(),
            self.0[1].make_checkpoint(),
            self.0[2].make_checkpoint(),
        )?;
        Ok([r0, r1, r2])
    }

    /// Seed an archival genesis checkpoint for all 3 parties concurrently.
    ///
    /// Identical to [`seed_all`] but marks the inserted row as archival (`is_archival = true`).
    /// Archival checkpoints are preserved by [`PruningMode::OlderNonArchival`] and are only
    /// removed by [`PruningMode::AllOlder`].
    pub async fn make_archival_checkpoints(&self) -> eyre::Result<[GraphCheckpointRow; 3]> {
        let (r0, r1, r2) = tokio::try_join!(
            self.0[0].make_archival_checkpoint(),
            self.0[1].make_archival_checkpoint(),
            self.0[2].make_archival_checkpoint(),
        )?;
        Ok([r0, r1, r2])
    }

    /// Delete all S3 objects from all checkpoint buckets.
    ///
    /// Iterates through all parties' checkpoint buckets, lists all objects,
    /// and deletes them. Ignores individual deletion errors to handle cases
    /// where objects may have already been deleted.
    pub async fn clear_all_s3_buckets(&self, configs: &CpuConfigs) -> eyre::Result<()> {
        for (node, config) in self.0.iter().zip(configs.iter()) {
            let bucket = &config.checkpoint_bucket;
            let keys = node.list_s3_keys(bucket).await?;
            for key in keys {
                // Ignore errors — objects may already be gone or other transient issues.
                let _ = node
                    .s3
                    .delete_object()
                    .bucket(bucket)
                    .key(&key)
                    .send()
                    .await;
            }
        }
        Ok(())
    }

    /// Assert all parties produced the same checkpoint hash and that it matches
    /// the WAL-materialized reference graph from party 0.
    pub async fn assert_consensus_and_reference(&self) -> eyre::Result<()> {
        self.assert_checkpoint_hashes_agree().await?;
        let reference_hash = self.0[0].store.compute_reference_hash().await?;
        self.assert_checkpoint_hashes_match_reference(&reference_hash)
            .await
    }

    /// Apply the same assertions to all three parties.
    pub async fn apply_uniform_assertions(&self, assertions: &WalAssertions) -> eyre::Result<()> {
        self.apply_assertions(&[assertions.clone(), assertions.clone(), assertions.clone()])
            .await
    }

    /// Apply one assertion to party 0 and a different one to parties 1 and 2.
    /// Useful for tests that desync or stagger party 0.
    pub async fn apply_split_assertions(
        &self,
        party0: &WalAssertions,
        parties_12: &WalAssertions,
    ) -> eyre::Result<()> {
        self.apply_assertions(&[party0.clone(), parties_12.clone(), parties_12.clone()])
            .await
    }
}

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// Post-condition assertions for a single party's WAL and checkpoint state.
///
/// All fields are optional — only set fields are checked.
///
/// S3 checkpoint object existence is verified automatically for all checkpoints
/// whenever assertions are run. When checkpoint_count is asserted, the S3 object
/// count is also automatically verified to match.
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
}
