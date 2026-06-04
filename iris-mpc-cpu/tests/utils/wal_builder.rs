use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::{
        graph_store::GraphPg,
        mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint},
    },
};
use iris_mpc_utils::{aws::AwsClient, irises::generate_iris_shares_for_upload_both_eyes};
use rand::{rngs::StdRng, SeedableRng};

use super::cpu_node::CpuNodes;

/// Builds and inserts synthetic `hawk_graph_mutations` rows without requiring a live
/// MPC request pipeline or real iris data.
///
/// Each inserted row corresponds to one `modification_id` and contains a
/// bincode-serialized `BothEyes<Vec<GraphMutation<IrisVectorId>>>`.
///
/// ## Constraints
///
/// - `AddNode` and `AddEdges` are both safe.  Reset-update, recovery-update, and
///   other modification types assume a node already exists and are not supported here.
/// - Use sequential `node_id` values (0, 1, 2, …).
/// - Keep neighbor lists under 100 entries per `add_edges` call to stay within
///   realistic HNSW degree bounds.
///
/// ## Usage
///
/// ```rust
/// WalMutationBuilder::new()
///     .add_node(1, 0, 3)                     // mod_id=1, node_id=0, height=3
///     .add_node(2, 1, 2)                     // mod_id=2, node_id=1, height=2
///     .add_edges(3, 0, vec![1], 0)           // mod_id=3, base=0, neighbors=[1], layer=0
///     .build(&nodes)
///     .await?;
/// ```
pub struct WalMutationBuilder {
    entries: Vec<(i64, GraphMutation<IrisVectorId>)>,
}

impl WalMutationBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an `AddNode` mutation for both eyes at the given `modification_id`.
    ///
    /// `node_id` is the HNSW node identifier (use sequential values starting from 0).
    /// `height` is the HNSW layer height for this node.
    pub fn add_node(mut self, modification_id: i64, node_id: u32, height: usize) -> Self {
        let mutation = GraphMutation {
            seq_no: modification_id as u64,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(node_id),
                height,
                update_ep: UpdateEntryPoint::False,
            }],
        };
        self.entries.push((modification_id, mutation));
        self
    }

    /// Add an `AddEdges` mutation for both eyes at the given `modification_id`.
    ///
    /// `base` is the source node; `neighbors` are the nodes it connects to at `layer`.
    /// Keep `neighbors.len()` under 100 to stay within realistic HNSW degree bounds.
    pub fn add_edges(
        mut self,
        modification_id: i64,
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
    ) -> Self {
        assert!(
            neighbors.len() <= 100,
            "neighbors.len() = {} exceeds the 100-edge limit per add_edges call",
            neighbors.len()
        );
        let mutation = GraphMutation {
            seq_no: modification_id as u64,
            ops: vec![MutationOp::AddEdges {
                base: IrisVectorId::from_serial_id(base),
                neighbors: neighbors
                    .iter()
                    .map(|&n| IrisVectorId::from_serial_id(n))
                    .collect(),
                layer,
                edge_type: EdgeType::All,
            }],
        };
        self.entries.push((modification_id, mutation));
        self
    }

    /// Add `count` sequential AddNode mutations. Node serial IDs are 0..count,
    /// mod IDs are 1..=count (i.e. mod_id = serial_id + 1).
    pub fn add_nodes_sequential(self, count: usize, height: u32) -> Self {
        self.add_nodes_sequential_from(1, count, height)
    }

    /// Add `count` sequential AddNode mutations starting at `start_mod_id`.
    /// Serial IDs start at 0 (relative to this batch): serial = idx, mod_id = start_mod_id + idx.
    /// Use this for WAL batches whose mod_ids are offset from 1 (e.g. 51..=100).
    pub fn add_nodes_sequential_from(self, start_mod_id: i64, count: usize, height: u32) -> Self {
        (0..count as i64).fold(self, |b, idx| {
            b.add_node(start_mod_id + idx, idx as u32, height as usize)
        })
    }

    /// Add wrapping AddEdges mutations for `count` nodes. Each node i gets edges
    /// to (i+1)%count and (i+2)%count. `edges_start_mod_id` is the mod_id for
    /// the first edge entry.
    pub fn add_edges_wrapping(self, count: usize, edges_start_mod_id: i64, layer: u32) -> Self {
        (0..count as i64).fold(self, |b, idx| {
            let base = idx as u32;
            let n = count as u32;
            b.add_edges(
                edges_start_mod_id + idx,
                base,
                vec![(base + 1) % n, (base + 2) % n],
                layer as usize,
            )
        })
    }

    /// Persist all mutations to one party's graph store.
    ///
    /// For each entry serializes `BothEyes<Vec<GraphMutation<IrisVectorId>>>` with
    /// bincode and calls `graph.upsert_hawk_graph_mutations(tx, mod_id, bytes)`.
    pub async fn insert_mutations(&self, graph: &GraphPg<PlaintextStore>) -> eyre::Result<()> {
        for (modification_id, mutation) in &self.entries {
            let both_eyes: BothEyes<Vec<GraphMutation<IrisVectorId>>> =
                [vec![mutation.clone()], vec![mutation.clone()]];
            let bytes = bincode::serialize(&both_eyes)?;
            let mut tx = graph.pool().begin().await?;
            graph
                .upsert_hawk_graph_mutations(&mut tx, *modification_id, &bytes)
                .await?;
            tx.commit().await?;
        }
        Ok(())
    }

    /// Convenience: seed the same mutations into all 3 parties' stores.
    pub async fn insert_mutations_all(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        for node in &nodes.0 {
            self.insert_mutations(&node.store.graph).await?;
        }
        Ok(())
    }

    /// Insert a `modifications` row for each WAL entry into one party's database.
    ///
    /// Each row is seeded as completed and persisted, reflecting the state of a
    /// fully-processed modification.  The `serial_id` is derived from the mutation
    /// op: `AddNode → node id`, `AddEdges → base node id`.
    ///
    /// `s3_url` is set to a deterministic UUID derived from the `modification_id`
    /// (no actual S3 upload is required — all rows are pre-persisted, so
    /// `hawk_main` will never attempt to fetch their iris shares).
    ///
    /// `result_message_body` mirrors the genesis pipeline format:
    /// `{"node_id": <party_id>}`.
    ///
    /// This mirrors what the genesis test pipeline writes via `db_ops::write_modification`,
    /// making workflow test state realistic enough for `hawk_main`'s modification sync
    /// to operate on a non-empty `modifications` table.
    pub async fn seed_modifications(
        &self,
        graph: &GraphPg<PlaintextStore>,
        party_id: usize,
    ) -> eyre::Result<()> {
        let result_message_body = format!(r#"{{"node_id":{party_id}}}"#);
        for (modification_id, mutation) in &self.entries {
            let serial_id: i64 = match mutation.ops.first() {
                Some(MutationOp::AddNode { id, .. }) => id.serial_id() as i64,
                Some(MutationOp::AddEdges { base, .. }) => base.serial_id() as i64,
                _ => 0,
            };
            // Deterministic UUID: no upload needed since all rows are persisted=TRUE.
            let s3_url = uuid::Uuid::from_u128(*modification_id as u128).to_string();
            sqlx::query(
                r#"
                INSERT INTO modifications
                    (id, serial_id, request_type, s3_url, status, persisted, result_message_body)
                VALUES ($1, $2, 'uniqueness', $3, 'COMPLETED', TRUE, $4)
                ON CONFLICT (id) DO NOTHING
                "#,
            )
            .bind(modification_id)
            .bind(serial_id)
            .bind(&s3_url)
            .bind(&result_message_body)
            .execute(graph.pool())
            .await?;
        }
        Ok(())
    }

    /// Convenience: seed modifications into all 3 parties' stores.
    ///
    /// Each party receives its own `result_message_body` with the correct `node_id`.
    /// All entries are seeded as `persisted = TRUE`.
    pub async fn seed_modifications_all(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        for node in &nodes.0 {
            self.seed_modifications(&node.store.graph, node.config.party_id)
                .await?;
        }
        Ok(())
    }

    /// Convenience: insert mutations and seed modifications into all 3 parties' stores.
    ///
    /// Equivalent to calling `insert_mutations_all` followed by `seed_modifications_all`.
    pub async fn build(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        self.insert_mutations_all(nodes).await?;
        self.seed_modifications_all(nodes).await?;
        Ok(())
    }

    /// Insert a `modifications` row for each WAL entry with a per-party persisted count.
    ///
    /// The first `persisted_count` entries (by insertion order) are seeded as
    /// `persisted = TRUE`; the remainder as `persisted = FALSE`.  This mirrors
    /// the staggered state across parties in the genesis modification-sync test:
    ///
    /// ```text
    /// party 0: persisted_count = 0  → all FALSE
    /// party 1: persisted_count = 5  → first 5 TRUE, rest FALSE
    /// party 2: persisted_count = 10 → all TRUE
    /// ```
    pub async fn seed_modifications_partial(
        &self,
        graph: &GraphPg<PlaintextStore>,
        party_id: usize,
        persisted_count: usize,
    ) -> eyre::Result<()> {
        let result_message_body = format!(r#"{{"node_id":{party_id}}}"#);
        for (idx, (modification_id, mutation)) in self.entries.iter().enumerate() {
            let serial_id: i64 = match mutation.ops.first() {
                Some(MutationOp::AddNode { id, .. }) => id.serial_id() as i64,
                Some(MutationOp::AddEdges { base, .. }) => base.serial_id() as i64,
                _ => 0,
            };
            let s3_url = uuid::Uuid::from_u128(*modification_id as u128).to_string();
            let persisted = idx < persisted_count;
            sqlx::query(
                r#"
                INSERT INTO modifications
                    (id, serial_id, request_type, s3_url, status, persisted, result_message_body)
                VALUES ($1, $2, 'uniqueness', $3, 'COMPLETED', $4, $5)
                ON CONFLICT (id) DO NOTHING
                "#,
            )
            .bind(modification_id)
            .bind(serial_id)
            .bind(&s3_url)
            .bind(persisted)
            .bind(&result_message_body)
            .execute(graph.pool())
            .await?;
        }
        Ok(())
    }

    /// Upload fake iris shares to S3 for each modification entry.
    ///
    /// Each entry's S3 key is the deterministic UUID derived from `modification_id`,
    /// matching the `s3_url` written by `seed_modifications` / `seed_modifications_partial`.
    /// This is required when the modification `request_type` is `'uniqueness'` (or any
    /// other type that triggers an S3 fetch in `sync_modifications`).
    ///
    /// The `aws_client` must already have its public keyset loaded
    /// (`client.set_public_keyset().await?` called before passing it here).
    ///
    /// Shares are randomly generated — their iris content does not matter for
    /// WAL-sync tests; only the presence of a valid S3 object at the expected key
    /// is required.
    pub async fn upload_iris_shares(&self, aws_client: &AwsClient) -> eyre::Result<()> {
        let mut rng = StdRng::seed_from_u64(42);
        for (modification_id, _) in &self.entries {
            let uuid = uuid::Uuid::from_u128(*modification_id as u128);
            let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
            aws_client
                .s3_upload_iris_shares(&uuid, &shares)
                .await
                .map_err(|e| {
                    eyre::eyre!("S3 upload failed for modification_id={modification_id}: {e}")
                })?;
        }
        Ok(())
    }
}

impl Default for WalMutationBuilder {
    fn default() -> Self {
        Self::new()
    }
}
