use crate::utils::CpuNodeConfig;

use super::CpuConfigs;

use eyre::eyre;
use iris_mpc_common::{
    postgres::{run_migrations, AccessMode, PostgresClient},
    MASK_CODE_LENGTH,
};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::graph::graph_store::{GraphCheckpointRow, GraphPg};
use iris_mpc_cpu::hnsw::GraphMem;
use iris_mpc_cpu::utils::serialization::graph::GraphFormat;
use iris_mpc_cpu::utils::serialization::types::graph_v3;
use iris_mpc_cpu::{
    graph_checkpoint::stream_serialize_and_upload_with,
    utils::serialization::types::graph_v3::write_graph_v3,
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use iris_mpc_utils::{aws::AwsClient, irises::generate_iris_shares_for_upload_both_eyes};
use rand::{rngs::StdRng, SeedableRng};

/// Per-party database handles for test setup and post-condition assertions.
///
/// The checkpoint format (`bincode([GraphMem; 2])`) is store-agnostic, so `PlaintextStore`
/// is safe to use here even though `hawk_main` uses `Aby3Store` at runtime.
pub struct DbStores {
    pub graph: GraphPg<PlaintextStore>,
    pub iris_store: IrisStore,
}

impl DbStores {
    pub async fn new(config: &CpuNodeConfig) -> eyre::Result<Self> {
        let pg_client =
            PostgresClient::new(&config.db_url, &config.db_schema, AccessMode::ReadWrite).await?;
        run_migrations(&pg_client.pool, false).await?;
        let graph = GraphPg::new(&pg_client).await?;
        let iris_store = IrisStore::new(&pg_client).await?;
        Ok(Self { graph, iris_store })
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

    /// Truncate all WAL, checkpoint, and iris tables for this party.
    /// Covers: hawk_graph_mutations, genesis_graph_checkpoint, persistent_state, modifications, and irises.
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
        sqlx::query("TRUNCATE irises")
            .execute(&self.iris_store.pool)
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
    /// Returns the s3_key of the deleted checkpoint, or None if no checkpoint existed.
    ///
    /// Intended for checkpoint-desync tests that need to artificially regress one
    /// party's checkpoint state.
    pub async fn delete_latest_checkpoint(&self) -> eyre::Result<Option<String>> {
        // First fetch the latest checkpoint to get its s3_key before deletion
        if let Some(checkpoint) = self.latest_checkpoint().await? {
            sqlx::query(
                "DELETE FROM genesis_graph_checkpoint \
                 WHERE id = (SELECT MAX(id) FROM genesis_graph_checkpoint)",
            )
            .execute(self.graph.pool())
            .await?;
            Ok(Some(checkpoint.s3_key))
        } else {
            Ok(None)
        }
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
    /// `bincode::serialize_into(writer, &graph)` where `graph: BothEyes<GraphMem>`.
    /// Use this to build a reference hash for `CpuNodes::assert_checkpoint_hashes_match_reference`.
    pub async fn compute_reference_hash(&self) -> eyre::Result<[u8; 32]> {
        let rows = self.graph.get_hawk_graph_mutations_after(None).await?;
        let mut graph_pair: [GraphMem; 2] = [GraphMem::new(), GraphMem::new()];
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
    pub async fn new(config: CpuNodeConfig, s3: aws_sdk_s3::Client) -> eyre::Result<Self> {
        let stores = DbStores::new(&config).await?;
        Ok(Self {
            store: stores,
            s3,
            config,
        })
    }

    /// Insert one iris share into this node's iris store.
    ///
    /// Writes the party-specific left/right code+mask into the `irises` table
    /// using the node's own schema, so the `check_store_consistency` invariant
    /// (`COUNT(*) == MAX(id)`) holds when the server starts.
    pub async fn insert_iris_share(
        &self,
        id: i64,
        left_code: &[u16],
        left_mask: &[u16],
        right_code: &[u16],
        right_mask: &[u16],
    ) -> eyre::Result<()> {
        let mut tx = self.store.iris_store.pool.begin().await?;
        self.store
            .iris_store
            .insert_irises(
                &mut tx,
                &[StoredIrisRef {
                    id,
                    left_code,
                    left_mask,
                    right_code,
                    right_mask,
                }],
            )
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Delete the latest checkpoint from both the database and S3.
    ///
    /// This ensures that when a checkpoint row is removed from the DB, its
    /// corresponding S3 object is also cleaned up to maintain consistency.
    pub async fn delete_latest_checkpoint(&self) -> eyre::Result<()> {
        if let Some(s3_key) = self.store.delete_latest_checkpoint().await? {
            // Best-effort cleanup: ignore errors if the S3 object is already gone
            let _ = self
                .s3
                .delete_object()
                .bucket(&self.config.checkpoint_bucket)
                .key(&s3_key)
                .send()
                .await;
        }
        Ok(())
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

    pub async fn make_checkpoint_v3(&self) -> eyre::Result<GraphCheckpointRow> {
        self.make_checkpoint_inner(false, GraphFormat::V3).await
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
            // Filter S3 keys by party_id since the bucket is shared across all parties.
            let all_keys = self.list_s3_keys(&self.config.checkpoint_bucket).await?;
            let party_id_str = format!("{}/", self.config.party_id);
            let party_keys: Vec<String> = all_keys
                .iter()
                .filter(|k| {
                    k.starts_with(&party_id_str)
                        || k.starts_with(&format!("genesis/{}/", self.config.party_id))
                })
                .cloned()
                .collect();
            let s3_count = party_keys.len();
            eyre::ensure!(
                s3_count == expected,
                "S3 object count in bucket {} for party {}: expected {expected}, got {s3_count}",
                self.config.checkpoint_bucket,
                self.config.party_id
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

    async fn make_checkpoint_with_options(
        &self,
        is_archival: bool,
    ) -> eyre::Result<GraphCheckpointRow> {
        self.make_checkpoint_inner(is_archival, GraphFormat::Current)
            .await
    }

    /// Build and persist a genesis checkpoint for this party.
    ///
    /// Replays all WAL mutations from the DB to construct the graph state, then
    /// serializes according to `format` and uploads to S3. The most recent
    /// `modification_id` is looked up automatically and used as both
    /// `last_indexed_iris_id` and `last_indexed_modification_id`.
    ///
    /// 1. Query `MAX(modification_id)` from `hawk_graph_mutations`.
    /// 2. Read all `hawk_graph_mutations` rows up to that id.
    /// 3. Replay mutations into `[GraphMem; 2]` (left + right eye).
    /// 4. Serialize according to `format`:
    ///    - `Current` → `bincode::serialize(&[GraphMem; 2])` (sidecar wire format).
    ///    - `V3`      → deterministic custom encoding via `write_graph_v3` (no `seq_no`).
    /// 5. Compute BLAKE3 hash of the bytes.
    /// 6. Upload to S3 at key `"{party_id}/checkpoints/{label}_{last_modification_id}"`,
    ///    where `label` is `"archival"` when `is_archival` is true, otherwise `"seed"`.
    /// 7. Insert `genesis_graph_checkpoint` DB row inside a transaction.
    /// 8. Return the newly inserted `GraphCheckpointRow`.
    async fn make_checkpoint_inner(
        &self,
        is_archival: bool,
        format: GraphFormat,
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

        let mutation_rows = self
            .store
            .graph
            .get_hawk_graph_mutations(Some(last_modification_id))
            .await?;

        let mut left_graph = GraphMem::new();
        let mut right_graph = GraphMem::new();
        for row in &mutation_rows {
            let both_eyes = row.deserialize_mutations()?;
            left_graph.insert_apply_all(&both_eyes[0])?;
            right_graph.insert_apply_all(&both_eyes[1])?;
        }

        let bytes: Vec<u8> = match &format {
            GraphFormat::Current => {
                // bincode-serialised [GraphMem; 2] — matches the sidecar wire format.
                let graph_pair: [GraphMem; 2] = [left_graph, right_graph];
                bincode::serialize(&graph_pair)?
            }
            GraphFormat::V3 => {
                // Convert GraphMem → GraphV3 by copying entry_points and layers,
                // dropping the seq_no field that is absent in the V3 format.
                let to_v3 = |g: GraphMem| -> graph_v3::GraphV3 {
                    let vec_id = |v: u32| graph_v3::VectorId { id: v, version: 0 };
                    graph_v3::GraphV3 {
                        entry_point: g
                            .entry_points
                            .iter()
                            .map(|ep| graph_v3::EntryPoint {
                                point: vec_id(ep.point),
                                layer: ep.layer,
                            })
                            .collect(),
                        layers: g
                            .layers
                            .iter()
                            .map(|layer| graph_v3::Layer {
                                links: layer
                                    .links
                                    .iter()
                                    .map(|(k, vs)| {
                                        (
                                            vec_id(*k),
                                            graph_v3::EdgeIds(
                                                vs.neighbors().iter().map(|&n| vec_id(n)).collect(),
                                            ),
                                        )
                                    })
                                    .collect(),
                                set_hash: layer.checksum(),
                            })
                            .collect(),
                    }
                };
                // Use deterministic serialization so all parties produce identical bytes
                // for the same logical graph (HashMap iteration order is non-deterministic).
                let graph_pair = [to_v3(left_graph), to_v3(right_graph)];
                let mut bytes = Vec::new();
                write_graph_v3(&mut bytes, &graph_pair[0])?;
                write_graph_v3(&mut bytes, &graph_pair[1])?;
                bytes
            }
            _ => unimplemented!(),
        };

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
            format.version(),
        )
        .await?;
        tx.commit().await?;

        self.store
            .graph
            .get_latest_genesis_graph_checkpoint()
            .await?
            .ok_or_else(|| eyre!("no checkpoint row found after insert"))
    }
}

/// All three parties' nodes.  Mirrors `MpcNodes` from the genesis tests.
pub struct CpuNodes(pub [CpuNode; 3]);

impl CpuNodes {
    pub async fn new(configs: &CpuConfigs, s3: aws_sdk_s3::Client) -> eyre::Result<Self> {
        // Construct all 3 concurrently.
        let (n0, n1, n2) = tokio::try_join!(
            CpuNode::new(configs[0].clone(), s3.clone()),
            CpuNode::new(configs[1].clone(), s3.clone()),
            CpuNode::new(configs[2].clone(), s3),
        )?;
        Ok(Self([n0, n1, n2]))
    }

    /// Create nodes and immediately truncate checkpoint tables and clear S3 buckets —
    /// the standard starting state for every workflow test.
    pub async fn new_clean(configs: &CpuConfigs, s3: aws_sdk_s3::Client) -> eyre::Result<Self> {
        let nodes = Self::new(configs, s3).await?;
        nodes.clear_all_s3_buckets(configs).await?;
        nodes.truncate_checkpoint_tables().await?;
        // Log the exclusions file if it exists — leftover state from a prior run
        // may or may not affect this test, but it is useful to know about.
        if let Ok(resp) = nodes.0[0]
            .s3
            .get_object()
            .bucket("wf-smpcv2-dev-sync-protocol")
            .key("dev_deleted_serial_ids.json")
            .send()
            .await
        {
            if let Ok(body) = resp.body.collect().await {
                let bytes = body.into_bytes();
                let text = String::from_utf8_lossy(&bytes);
                tracing::warn!(
                    exclusions = %text,
                    "dev_deleted_serial_ids.json exists in S3 at test start"
                );
            }
        }
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

    /// Seed a V3-format genesis checkpoint for all 3 parties concurrently.
    ///
    /// Each checkpoint is non-archival and serialized without the `seq_no`
    /// field (see [`CpuNode::make_checkpoint_v3`]).
    pub async fn make_checkpoints_v3(&self) -> eyre::Result<[GraphCheckpointRow; 3]> {
        let (r0, r1, r2) = tokio::try_join!(
            self.0[0].make_checkpoint_v3(),
            self.0[1].make_checkpoint_v3(),
            self.0[2].make_checkpoint_v3(),
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

    /// Upload deterministic fake iris shares to S3 for modifications 1..=count,
    /// then update every party's `modifications.s3_url` to the uploaded key.
    ///
    /// All parties receive the same S3 key (technically incorrect for real MPC,
    /// but sufficient for tests that only need the server to find an object at
    /// the expected key). Each share set is seeded by its modification ID so
    /// the content is stable across calls and test runs.
    pub async fn init_iris_shares(&self, count: usize, aws_client: &AwsClient) -> eyre::Result<()> {
        for modification_id in 1..=(count as i64) {
            let uuid = uuid::Uuid::from_u128(modification_id as u128);
            let mut rng = StdRng::seed_from_u64(modification_id as u64);
            let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
            aws_client
                .s3_upload_iris_shares(&uuid, &shares)
                .await
                .map_err(|e| {
                    eyre::eyre!("S3 upload failed for modification_id={modification_id}: {e}")
                })?;
            let s3_key = uuid.to_string();
            for node in &self.0 {
                sqlx::query("UPDATE modifications SET s3_url = $1 WHERE id = $2")
                    .bind(&s3_key)
                    .bind(modification_id)
                    .execute(node.store.graph.pool())
                    .await?;

                let party_id = node.config.party_id;
                let left = &shares[0][party_id];
                let right = &shares[1][party_id];
                node.insert_iris_share(
                    modification_id,
                    &left.code.coefs,
                    &left.mask.coefs[..MASK_CODE_LENGTH],
                    &right.code.coefs,
                    &right.mask.coefs[..MASK_CODE_LENGTH],
                )
                .await?;
            }
        }
        Ok(())
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
