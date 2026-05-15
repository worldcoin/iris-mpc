//! [`TerminalAction`] implementations for the checkpoint protocol.
//!
//! Two v0 actions:
//!
//! - [`UploadAndRecord`] serializes the materialized graph, uploads it to
//!   S3, and records a `genesis_graph_checkpoint` row. Used by the
//!   sidecar daemon and by in-Hawk background checkpointing.
//!
//! - [`InstallAsServing`] swaps the materialized graph into the live
//!   in-Hawk graph reference. Used by the Hawk restart path.

use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::checkpoint_protocol::{
    Blake3Hash, CheckpointMeta, CycleError, GraphSnapshot, TerminalAction,
};
use crate::execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT};
use crate::graph_checkpoint::upload_graph_checkpoint;
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};
use iris_mpc_common::vector_id::VectorId;

/// Terminal action that uploads the materialized graph to S3 and inserts a
/// new `genesis_graph_checkpoint` row.
///
/// `last_indexed_iris_id` is carried forward from `base` — Hawk Main does
/// not maintain per-mutation iris-id tracking, and the field is
/// observability-only in this code path. `last_indexed_modification_id`
/// and `graph_mutation_id` are both set to `snapshot.actual_height`.
pub struct UploadAndRecord<'a, V: VectorStore> {
    pub graph_store: &'a GraphPg<V>,
    pub s3_client: &'a S3Client,
    pub bucket: String,
    pub party_id: usize,
    pub is_archival: bool,
}

impl<'a, V: VectorStore + Send + Sync> UploadAndRecord<'a, V> {
    pub fn new(
        graph_store: &'a GraphPg<V>,
        s3_client: &'a S3Client,
        bucket: String,
        party_id: usize,
        is_archival: bool,
    ) -> Self {
        Self {
            graph_store,
            s3_client,
            bucket,
            party_id,
            is_archival,
        }
    }
}

#[async_trait]
impl<V: VectorStore + Send + Sync> TerminalAction for UploadAndRecord<'_, V> {
    async fn finalize(
        &mut self,
        base: CheckpointMeta,
        snapshot: GraphSnapshot,
        hash: Blake3Hash,
    ) -> Result<(), CycleError> {
        // upload_graph_checkpoint takes `&BothEyes<GraphRef>` (Arc<RwLock>).
        // We own the snapshot graph; wrap each eye to match the signature.
        // The wrappers are dropped after upload — no lasting allocation.
        let [left, right] = snapshot.graph;
        let wrapped: BothEyes<GraphRef> =
            [Arc::new(RwLock::new(left)), Arc::new(RwLock::new(right))];

        let last_indexed_iris_id_u32: u32 = base.last_indexed_iris_id.try_into().map_err(|_| {
            CycleError::Fatal(format!(
                "carried last_indexed_iris_id={} overflows u32",
                base.last_indexed_iris_id,
            ))
        })?;

        let state = upload_graph_checkpoint(
            &self.bucket,
            self.party_id,
            &wrapped,
            self.s3_client,
            last_indexed_iris_id_u32,
            snapshot.actual_height,
            Some(snapshot.actual_height),
            self.is_archival,
        )
        .await
        .map_err(|e| CycleError::Fatal(format!("upload_graph_checkpoint: {e}")))?;

        // Defensive check: the protocol's consensus hash and the upload's
        // recomputed hash must agree. Equality is structural today (both
        // bincode-serialize the same BothEyes<GraphMem<VectorId>>), but if
        // either serializer ever drifts, every subsequent cycle starting
        // from this checkpoint would silently fail base or hash agreement.
        // Catch the drift at the writing party.
        let local_hex = crate::checkpoint_protocol::hex(&hash);
        if local_hex != state.blake3_hash {
            return Err(CycleError::Fatal(format!(
                "consensus/storage hash mismatch: local={local_hex} stored={}",
                state.blake3_hash,
            )));
        }

        let mut tx = self
            .graph_store
            .tx()
            .await
            .map_err(|e| CycleError::Transient(format!("begin tx: {e}")))?;
        GraphPg::<V>::insert_genesis_graph_checkpoint(
            &mut tx.tx,
            &state.s3_key,
            state.last_indexed_iris_id as i64,
            state.last_indexed_modification_id,
            state.graph_mutation_id,
            &state.blake3_hash,
            self.is_archival,
            state.graph_version,
        )
        .await
        .map_err(|e| CycleError::Transient(format!("insert_genesis_graph_checkpoint: {e}")))?;
        tx.tx
            .commit()
            .await
            .map_err(|e| CycleError::Transient(format!("commit tx: {e}")))?;

        Ok(())
    }
}

/// Terminal action that swaps the materialized graph into the live in-Hawk
/// graph reference. Used at startup so the first request hits the rebuilt
/// graph; the protocol's hash consensus has already proven the graph is
/// byte-identical across parties.
pub struct InstallAsServing {
    pub target: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>,
}

impl InstallAsServing {
    pub fn new(target: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>) -> Self {
        Self { target }
    }
}

#[async_trait]
impl TerminalAction for InstallAsServing {
    async fn finalize(
        &mut self,
        _base: CheckpointMeta,
        snapshot: GraphSnapshot,
        _hash: Blake3Hash,
    ) -> Result<(), CycleError> {
        let [snap_left, snap_right] = snapshot.graph;
        *self.target[LEFT].write().await = snap_left;
        *self.target[RIGHT].write().await = snap_right;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_protocol::FreezeHeight;
    use crate::hnsw::graph::mutation::{GraphMutation, UpdateEntryPoint};

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    fn cp_meta() -> CheckpointMeta {
        CheckpointMeta {
            checkpoint_id: 1,
            s3_key: "cp/1".into(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 0,
            graph_mutation_id: None,
            blake3_hash: "0".into(),
            graph_version: 1,
        }
    }

    /// `InstallAsServing` replaces the target's contents wholesale; pre-existing
    /// graph state in the target is discarded.
    #[tokio::test]
    async fn install_as_serving_swaps_in_snapshot_graph() {
        // Target holds an existing graph with node 99.
        let target: BothEyes<Arc<RwLock<GraphMem<VectorId>>>> = [
            Arc::new(RwLock::new(GraphMem::new())),
            Arc::new(RwLock::new(GraphMem::new())),
        ];
        for eye in &target {
            let mut g = eye.write().await;
            g.insert_apply(vec![GraphMutation::AddNode {
                id: vid(99),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }]);
        }

        // Snapshot carries a different graph (node 1).
        let snapshot = {
            let mut left = GraphMem::new();
            let mut right = GraphMem::new();
            for g in [&mut left, &mut right] {
                g.insert_apply(vec![GraphMutation::AddNode {
                    id: vid(1),
                    height: 1,
                    update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
                }]);
            }
            GraphSnapshot {
                graph: [left, right],
                actual_height: FreezeHeight(42).0,
            }
        };

        let mut t = InstallAsServing::new([Arc::clone(&target[0]), Arc::clone(&target[1])]);
        t.finalize(cp_meta(), snapshot, [0u8; 32]).await.unwrap();

        // After finalize, the target has node 1 and no longer has node 99.
        for eye in &target {
            let g = eye.read().await;
            let has_1 = g
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(1)).is_some());
            let has_99 = g
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(99)).is_some());
            assert!(has_1, "snapshot's vid(1) should be installed");
            assert!(!has_99, "target's prior vid(99) should be gone");
        }
    }

    /// The protocol's consensus hash (computed by [`Blake3GraphHasher`])
    /// MUST equal the hash that `upload_graph_checkpoint` recomputes and
    /// stores in `genesis_graph_checkpoint.blake3_hash`. Both paths
    /// bincode-serialize the same `BothEyes<GraphMem<VectorId>>`, but
    /// through different argument shapes:
    ///
    /// - Hasher: `bincode::serialize_into(blake3, &graph)` where
    ///   `graph: &[GraphMem; 2]`.
    /// - Upload: `bincode::serialize(&[&*left, &*right])` where the
    ///   argument is `&[&GraphMem; 2]`.
    ///
    /// If they ever diverge, every cycle from the resulting checkpoint
    /// fails base or hash agreement silently. `UploadAndRecord::finalize`
    /// has a runtime mismatch check; this test pins the property down at
    /// the serializer level so any drift surfaces in unit tests.
    #[test]
    fn consensus_hash_matches_upload_path_serialization() {
        use crate::checkpoint_protocol::Blake3GraphHasher;
        use crate::checkpoint_protocol::GraphHasher;
        use crate::execution::hawk_main::{LEFT, RIGHT};

        // Build a non-trivial graph: distinct content per eye so eye-order
        // swaps would fail the assertion loudly.
        let mut left = GraphMem::<VectorId>::new();
        let mut right = GraphMem::<VectorId>::new();
        left.insert_apply(vec![GraphMutation::AddNode {
            id: vid(7),
            height: 2,
            update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
        }]);
        right.insert_apply(vec![GraphMutation::AddNode {
            id: vid(11),
            height: 1,
            update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
        }]);
        let graph = [left, right];

        // Consensus path.
        let consensus_hash = Blake3GraphHasher::new().hash_canonical(&graph);

        // Upload path (the inner bincode + blake3 sequence from
        // upload_graph_checkpoint, replicated without S3 I/O).
        let upload_bytes =
            bincode::serialize(&[&graph[LEFT], &graph[RIGHT]]).expect("bincode serialize");
        let upload_hash: [u8; 32] = *blake3::hash(&upload_bytes).as_bytes();

        assert_eq!(
            consensus_hash, upload_hash,
            "Blake3GraphHasher and the upload path must produce byte-identical hashes; \
             if you changed one serializer, change both (or refactor to share)."
        );
    }
}
