use super::{cpu_node::CpuNodes, CpuConfigs};

// TODO (open question #3): confirm insert method name and signature.
// use iris_mpc_cpu::hnsw::graph::graph_store::{GraphPg, GraphCheckpointRow};
// use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;

// Types needed for checkpoint construction — confirmed store-agnostic:
// use iris_mpc_cpu::hnsw::graph::GraphMem;
// use iris_mpc_cpu::utils::serialization::types::graph_v4::GraphV4;
// use iris_mpc_cpu::graph_checkpoint::s3_client::streaming_upload::stream_serialize_and_upload_with;

/// Pre-seeds a base checkpoint into both the DB and S3 for a single party.
///
/// The checkpoint format is `GraphV4` serialized via bincode — confirmed store-agnostic.
/// The same blob is accepted by `hawk_main` on startup regardless of whether it was
/// produced by a `PlaintextStore`-backed test or the real `Aby3Store`-backed service.
///
/// After seeding, the caller uses `WalMutationBuilder` to add WAL mutations with
/// `modification_id > last_modification_id` — those will be rolled forward on startup.
///
/// ## Pipeline per party
///
/// 1. Create empty `GraphMem<IrisVectorId>` (no nodes, zero entry point)
/// 2. Convert to `GraphV4` via `From<GraphMem<IrisVectorId>>`
/// 3. Serialize with `bincode::serialize`
/// 4. Compute BLAKE3 hash of the bytes
/// 5. Upload bytes to S3 at key `"{party_id}/checkpoints/seed_{last_mod_id}"`
/// 6. Insert `genesis_graph_checkpoint` DB row
pub struct CheckpointSeeder {
    /// Value stored as `last_indexed_iris_id` in the checkpoint row.
    pub last_iris_id: i64,
    /// WAL anchor: `hawk_main` replays mutations with `modification_id > last_modification_id`.
    pub last_modification_id: i64,
}

impl CheckpointSeeder {
    pub fn new(last_iris_id: i64, last_modification_id: i64) -> Self {
        Self { last_iris_id, last_modification_id }
    }

    /// Seed a checkpoint for one party.
    pub async fn seed_party(
        &self,
        _graph: &(),  // TODO: &GraphPg<PlaintextStore>
        _bucket: &str,
        _party_id: usize,
    ) -> eyre::Result<()> /* TODO: -> GraphCheckpointRow */ {
        // Step 1: Build an empty GraphMem.
        // TODO: let graph_mem = GraphMem::<IrisVectorId>::new();

        // Step 2: Convert to GraphV4.
        // TODO: let graph_v4 = GraphV4::from(graph_mem);

        // Step 3: Serialize with bincode.
        // TODO: let bytes = bincode::serialize(&graph_v4)?;

        // Step 4: Compute BLAKE3 hash.
        // TODO: let hash = blake3::hash(&bytes);
        // TODO: let hash_hex = hex::encode(hash.as_bytes());

        // Step 5: Upload to S3.
        // TODO: let s3_key = format!("{party_id}/checkpoints/seed_{}", self.last_modification_id);
        // TODO: upload bytes to S3 at s3_key using stream_serialize_and_upload_with or put_object.

        // Step 6: Insert DB row.
        // TODO (open question #3): call graph.insert_genesis_graph_checkpoint(
        //     self.last_iris_id,
        //     self.last_modification_id,
        //     &s3_key,
        //     &hash_hex,
        //     /* graph_version */ 4,
        //     /* is_archival */ false,
        // ).await

        todo!(
            "build empty GraphMem → GraphV4 → bincode → BLAKE3 → S3 upload → DB row insert"
        )
    }

    /// Seed the same checkpoint for all 3 parties.
    pub async fn seed_all(
        &self,
        _nodes: &CpuNodes,
        _configs: &CpuConfigs,
    ) -> eyre::Result<()> /* TODO: -> [GraphCheckpointRow; 3] */ {
        // TODO: call seed_party for each party with their respective graph store and bucket.
        // Can be parallelized with tokio::try_join!.
        todo!("seed checkpoint for all 3 parties")
    }
}
