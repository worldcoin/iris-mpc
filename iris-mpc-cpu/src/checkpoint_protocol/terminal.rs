use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;
use std::sync::Arc;
use tokio::sync::{oneshot, RwLock};

use crate::checkpoint_protocol::{
    Blake3Hash, CheckpointMeta, CycleError, FreezeHeight, Graph, TerminalAction,
};
use crate::execution::hawk_main::{BothEyes, LEFT, RIGHT};
use crate::graph_checkpoint::{
    cleanup_checkpoints, stream_serialize_and_upload_with, BlakeTeeWriter, GraphCheckpointState,
    PruningMode, DEFAULT_STREAMING_PARALLELISM, DEFAULT_STREAMING_PART_SIZE,
};
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};
use crate::utils::serialization::graph::GraphFormat;

/// Uploads the materialized graph to S3 and inserts a new
/// `genesis_graph_checkpoint` row.
///
/// `last_indexed_iris_id` is carried forward from `base` (Hawk Main doesn't
/// track per-mutation iris ids; the field is observability-only on this path).
pub struct UploadAndRecord<'a, V: VectorStore> {
    pub graph_store: &'a GraphPg<V>,
    pub s3_client: &'a S3Client,
    pub bucket: String,
    pub party_id: usize,
    pub is_archival: bool,
    pub pruning_mode: PruningMode,
}

impl<'a, V: VectorStore + Send + Sync> UploadAndRecord<'a, V> {
    pub fn new(
        graph_store: &'a GraphPg<V>,
        s3_client: &'a S3Client,
        bucket: String,
        party_id: usize,
        is_archival: bool,
        pruning_mode: PruningMode,
    ) -> Self {
        Self {
            graph_store,
            s3_client,
            bucket,
            party_id,
            is_archival,
            pruning_mode,
        }
    }
}

#[async_trait]
impl<V: VectorStore + Send + Sync> TerminalAction for UploadAndRecord<'_, V> {
    async fn finalize(
        &mut self,
        base: CheckpointMeta,
        freeze: FreezeHeight,
        graph: Graph,
        hash: Blake3Hash,
    ) -> Result<(), CycleError> {
        // Mirror the buffered upload's S3 key format so existing readers
        // (and the streaming download path) don't need a separate code path.
        let s3_key = format!(
            "genesis/{}/checkpoint_{}.bin",
            self.party_id,
            uuid::Uuid::new_v4()
        );

        // BlakeTee hashes wire bytes inline while bincode streams into the
        // multipart upload pipe — no fully serialized `Vec<u8>` exists. The
        // oneshot returns the digest from inside the closure.
        let (hash_tx, hash_rx) = oneshot::channel::<Blake3Hash>();
        stream_serialize_and_upload_with(
            self.s3_client,
            &self.bucket,
            &s3_key,
            move |w| {
                let mut tee = BlakeTeeWriter::new(w);
                bincode::serialize_into(&mut tee, &graph)
                    .map_err(|e| eyre::eyre!("bincode::serialize_into: {e}"))?;
                let _ = hash_tx.send(tee.finalize());
                Ok(())
            },
            DEFAULT_STREAMING_PART_SIZE,
            DEFAULT_STREAMING_PARALLELISM,
        )
        .await
        .map_err(|e| CycleError::Fatal(format!("stream_serialize_and_upload_with: {e}")))?;

        let upload_hash: Blake3Hash = hash_rx.await.map_err(|_| {
            CycleError::Fatal(
                "upload hash channel dropped before send — serializer closure did not run \
                 to completion despite Ok return"
                    .into(),
            )
        })?;

        // Guards against serializer drift between the hasher and upload paths.
        if upload_hash != hash {
            return Err(CycleError::Fatal(format!(
                "consensus/upload hash mismatch: consensus={} upload={}",
                hex::encode(hash),
                hex::encode(upload_hash),
            )));
        }
        let blake3_hash_hex = hex::encode(upload_hash);

        let mut tx = self
            .graph_store
            .tx()
            .await
            .map_err(|e| CycleError::Transient(format!("begin tx: {e}")))?;
        GraphPg::<V>::insert_genesis_graph_checkpoint(
            &mut tx.tx,
            &s3_key,
            base.last_indexed_iris_id,
            freeze.0,
            Some(freeze.0),
            &blake3_hash_hex,
            self.is_archival,
            GraphFormat::Current.version(),
        )
        .await
        .map_err(|e| CycleError::Transient(format!("insert_genesis_graph_checkpoint: {e}")))?;
        tx.tx
            .commit()
            .await
            .map_err(|e| CycleError::Transient(format!("commit tx: {e}")))?;

        let graph_checkpoint = GraphCheckpointState {
            s3_key,
            last_indexed_iris_id: base.last_indexed_iris_id as _,
            last_indexed_modification_id: freeze.0,
            graph_mutation_id: Some(freeze.0),
            blake3_hash: blake3_hash_hex,
            graph_version: GraphFormat::Current.version(),
            is_archival: self.is_archival,
        };
        // Retain the agreed base (and anything newer): peers provably hold
        // the base durably, but may not have recorded the new checkpoint yet.
        // See `cleanup_checkpoints` docs.
        if let Err(e) = cleanup_checkpoints(
            &self.bucket,
            self.s3_client,
            &graph_checkpoint,
            Some(base.checkpoint_id),
            self.graph_store,
            self.pruning_mode,
        )
        .await
        {
            tracing::warn!("failed to clean up old s3 checkpoints: {e}");
        }

        Ok(())
    }
}

/// Swaps the materialized graph into the live in-Hawk graph reference.
pub struct InstallAsServing {
    pub target: BothEyes<Arc<RwLock<GraphMem>>>,
}

impl InstallAsServing {
    pub fn new(target: BothEyes<Arc<RwLock<GraphMem>>>) -> Self {
        Self { target }
    }
}

#[async_trait]
impl TerminalAction for InstallAsServing {
    async fn finalize(
        &mut self,
        _base: CheckpointMeta,
        _freeze: FreezeHeight,
        graph: Graph,
        _hash: Blake3Hash,
    ) -> Result<(), CycleError> {
        let [left, right] = graph;
        *self.target[LEFT].write().await = left;
        *self.target[RIGHT].write().await = right;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
    use iris_mpc_common::VectorId;

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
        let target: BothEyes<Arc<RwLock<GraphMem>>> = [
            Arc::new(RwLock::new(GraphMem::new())),
            Arc::new(RwLock::new(GraphMem::new())),
        ];
        for eye in &target {
            let mut g = eye.write().await;
            g.insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![MutationOp::AddNode {
                    id: vid(99),
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 0 },
                }],
            })
            .unwrap();
        }

        // Snapshot graph carries a different graph (node 1).
        let snapshot_graph: Graph = {
            let mut left = GraphMem::new();
            let mut right = GraphMem::new();
            for g in [&mut left, &mut right] {
                g.insert_apply(&GraphMutation {
                    seq_no: 1,
                    ops: vec![MutationOp::AddNode {
                        id: vid(1),
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 0 },
                    }],
                })
                .unwrap();
            }
            [left, right]
        };

        let mut t = InstallAsServing::new([Arc::clone(&target[0]), Arc::clone(&target[1])]);
        t.finalize(cp_meta(), FreezeHeight(42), snapshot_graph, [0u8; 32])
            .await
            .unwrap();

        // After finalize, the target has node 1 and no longer has node 99.
        for eye in &target {
            let g = eye.read().await;
            let has_1 = g.get_layers().iter().any(|l| l.get_links(&1).is_some());
            let has_99 = g.get_layers().iter().any(|l| l.get_links(&99).is_some());
            assert!(has_1, "snapshot's vid(1) should be installed");
            assert!(!has_99, "target's prior vid(99) should be gone");
        }
    }

    /// Pins serializer-level equality between the consensus hash
    /// (`bincode::serialize_into(blake3, &[GraphMem; 2])`) and the upload-path
    /// hash (`bincode::serialize(&[&GraphMem; 2])`). Drift here silently breaks
    /// every cycle reading the resulting checkpoint.
    #[test]
    fn consensus_hash_matches_upload_path_serialization() {
        use crate::checkpoint_protocol::Blake3GraphHasher;
        use crate::checkpoint_protocol::GraphHasher;
        use crate::execution::hawk_main::{LEFT, RIGHT};

        // Distinct content per eye so any eye-order swap surfaces in the assertion.
        let mut left = GraphMem::new();
        let mut right = GraphMem::new();
        left.insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![MutationOp::AddNode {
                id: vid(7),
                height: 2,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        })
        .unwrap();
        right
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![MutationOp::AddNode {
                    id: vid(11),
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 0 },
                }],
            })
            .unwrap();
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
