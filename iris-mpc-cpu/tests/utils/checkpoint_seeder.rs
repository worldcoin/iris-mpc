use super::{cpu_node::CpuNodes, CpuConfigs};

// TODO: replace with real imports once open question #4 is resolved
// (PlaintextStore vs Aby3Store serialization compatibility).
// use iris_mpc_cpu::hnsw::graph::graph_store::{GraphPg, GraphCheckpointRow};
// use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;

/// Pre-seeds a base checkpoint into both the DB and S3 for a single party.
///
/// This establishes a starting point so that WAL roll-forward tests have a
/// realistic `(checkpoint, WAL delta)` state: after seeding, callers then use
/// `WalMutationBuilder` to add mutations with `modification_id > last_modification_id`.
///
/// # Open question #4
///
/// The serialized graph blob must be readable by `hawk_main` at startup, which uses
/// `GraphPg<Aby3Store>`.  If the checkpoint format is store-agnostic (raw HNSW
/// adjacency), a minimal empty-graph blob works.  If it embeds store-specific vector
/// data, this seeder needs to produce an Aby3-compatible blob.
pub struct CheckpointSeeder {
    /// Value to store as `last_indexed_iris_id` in the checkpoint row.
    pub last_iris_id: i64,
    /// WAL anchor: `hawk_main` will replay all mutations with
    /// `modification_id > last_modification_id` on startup.
    pub last_modification_id: i64,
}

impl CheckpointSeeder {
    pub fn new(last_iris_id: i64, last_modification_id: i64) -> Self {
        Self {
            last_iris_id,
            last_modification_id,
        }
    }

    /// Seed a checkpoint for one party: build minimal graph blob, upload to S3,
    /// insert `hawk_graph_checkpoints` row.
    ///
    /// Returns the inserted `GraphCheckpointRow` (placeholder type for now).
    pub async fn seed_party(
        &self,
        _graph: &(),     // TODO: &GraphPg<PlaintextStore>
        _s3: &(),        // TODO: &S3Client
        _bucket: &str,
        _party_id: usize,
    ) -> eyre::Result<()> /* TODO: -> GraphCheckpointRow */ {
        // TODO:
        //   1. build a minimal serialized Graph (empty node set, zero entry point)
        //      — must be compatible with what hawk_main deserializes on startup
        //   2. compute BLAKE3 hash of the serialized bytes
        //   3. upload bytes to S3 at key "{bucket}/{party_id}/checkpoints/seed"
        //   4. call graph.insert_genesis_graph_checkpoint(
        //          last_iris_id, last_modification_id, s3_key, blake3_hash, ...)
        //   5. return the inserted row
        todo!("seed checkpoint for one party")
    }

    /// Seed the same checkpoint for all 3 parties.
    pub async fn seed_all(
        &self,
        _nodes: &CpuNodes,
        _configs: &CpuConfigs,
    ) -> eyre::Result<[(); 3]> /* TODO: -> [GraphCheckpointRow; 3] */ {
        // TODO: call seed_party for each party with their respective S3 clients
        todo!("seed checkpoint for all parties")
    }
}
