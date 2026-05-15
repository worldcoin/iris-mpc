//! [`Materializer`] implementations for the checkpoint protocol.
//!
//! Two strategies in v0:
//!
//! - [`RebuildFromCheckpoint`] downloads the base checkpoint from S3 and
//!   replays WAL rows up to `freeze`. Deterministic; used by the sidecar
//!   daemon and by Hawk restart. Both eyes of each WAL row are applied
//!   atomically via [`GraphMem::insert_apply`], so left/right cannot drift.
//!
//! - [`LiveClone`] takes a read lock on the live in-Hawk graph and clones
//!   it. Used by an in-Hawk background checkpointing task. v0 clones the
//!   live state as-is and reports `actual_height = freeze.0`; the cloned
//!   graph's true WAL position is whatever the live graph happens to
//!   reflect at clone time. Determinism across parties relies on the
//!   subsequent hash-consensus round catching any drift, NOT on this
//!   materializer producing a graph at exactly `freeze`.

use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::checkpoint_protocol::{
    CheckpointMeta, CycleError, FreezeHeight, Graph, GraphSnapshot, Materializer, MutationStore,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::{download_graph_checkpoint, GraphCheckpointState};
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};
use futures::TryStreamExt;
use iris_mpc_common::vector_id::VectorId;

/// Materializer that rebuilds the graph deterministically from an S3
/// checkpoint plus WAL replay. Used by the sidecar daemon and by Hawk
/// restart.
pub struct RebuildFromCheckpoint<'a, V: VectorStore> {
    pub graph_store: &'a GraphPg<V>,
    pub s3_client: &'a S3Client,
    pub bucket: String,
}

impl<'a, V: VectorStore + Send + Sync> RebuildFromCheckpoint<'a, V> {
    pub fn new(graph_store: &'a GraphPg<V>, s3_client: &'a S3Client, bucket: String) -> Self {
        Self {
            graph_store,
            s3_client,
            bucket,
        }
    }
}

#[async_trait]
impl<V: VectorStore + Send + Sync> Materializer for RebuildFromCheckpoint<'_, V> {
    async fn snapshot(
        &mut self,
        base: CheckpointMeta,
        freeze: FreezeHeight,
    ) -> Result<GraphSnapshot, CycleError> {
        // Phase A: download the base graph from S3.
        let state = GraphCheckpointState {
            s3_key: base.s3_key.clone(),
            last_indexed_iris_id: base.last_indexed_iris_id.try_into().map_err(|_| {
                CycleError::Fatal(format!(
                    "checkpoint {} has invalid last_indexed_iris_id={}",
                    base.checkpoint_id, base.last_indexed_iris_id,
                ))
            })?,
            last_indexed_modification_id: base.last_indexed_modification_id,
            graph_mutation_id: base.graph_mutation_id,
            blake3_hash: base.blake3_hash.clone(),
            graph_version: base.graph_version,
            // Pruning policy is not protocol-relevant; the download path
            // only inspects this field via metrics. Default to false.
            is_archival: false,
        };

        // TODO: switch to the streaming download primitive from PR #2119 once it
        // lands in main — keeps peak RAM at ~part_size instead of the full graph
        // size during deserialize. Buffered path is correct, just memory-hungry.
        let mut graph: Graph =
            download_graph_checkpoint::<VectorId>(self.s3_client, &self.bucket, &state)
                .await
                .map_err(|e| {
                    CycleError::Fatal(format!(
                        "download_graph_checkpoint({}/{}): {e}",
                        self.bucket, state.s3_key,
                    ))
                })?;

        // Phase B: replay WAL rows in `(base.graph_mutation_id, freeze]`.
        let lo = base.graph_mutation_id.unwrap_or(0);
        let hi = freeze.0;
        if hi > lo {
            let stream = MutationStore::mutations_in_range(self.graph_store, lo, hi).await?;
            apply_wal_stream(&mut graph, stream).await?;
        }

        Ok(GraphSnapshot {
            graph,
            actual_height: hi,
        })
    }
}

/// Applies a WAL stream to an in-memory graph. Each stream item is one
/// `hawk_graph_mutations` row's deserialized payload — both eyes together —
/// so the left-eye and right-eye mutations of a row are applied as a unit
/// before advancing.
async fn apply_wal_stream<S>(graph: &mut Graph, stream: S) -> Result<(), CycleError>
where
    S: futures::Stream<
        Item = Result<
            BothEyes<Vec<crate::hnsw::graph::mutation::GraphMutation<VectorId>>>,
            CycleError,
        >,
    >,
{
    use crate::execution::hawk_main::{LEFT, RIGHT};
    tokio::pin!(stream);
    while let Some(row) = stream.try_next().await? {
        let [left_muts, right_muts] = row;
        graph[LEFT].insert_apply(left_muts);
        graph[RIGHT].insert_apply(right_muts);
    }
    Ok(())
}

/// Materializer that clones the in-memory live graph under a read lock.
/// Used by an in-Hawk background checkpointing task.
///
/// **v0 contract:** the returned graph reflects whatever WAL position the
/// live graph happens to be at when the lock is acquired. It is NOT
/// guaranteed to be exactly `freeze.0`. Cross-party determinism is enforced
/// by the subsequent hash-consensus round, not by this materializer.
pub struct LiveClone {
    pub graph: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>,
}

impl LiveClone {
    pub fn new(graph: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>) -> Self {
        Self { graph }
    }
}

#[async_trait]
impl Materializer for LiveClone {
    async fn snapshot(
        &mut self,
        _base: CheckpointMeta,
        freeze: FreezeHeight,
    ) -> Result<GraphSnapshot, CycleError> {
        let [left, right] = &self.graph;
        let left_clone = left.read().await.clone();
        let right_clone = right.read().await.clone();
        Ok(GraphSnapshot {
            graph: [left_clone, right_clone],
            actual_height: freeze.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_protocol::CheckpointMeta;
    use crate::execution::hawk_main::{LEFT, RIGHT};
    use crate::hnsw::graph::mutation::{GraphMutation, UpdateEntryPoint};
    use futures::stream;

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    fn add_node(n: u32) -> GraphMutation<VectorId> {
        GraphMutation::AddNode {
            id: vid(n),
            height: 1,
            update_ep: UpdateEntryPoint::False,
        }
    }

    fn row(
        left: Vec<u32>,
        right: Vec<u32>,
    ) -> Result<BothEyes<Vec<GraphMutation<VectorId>>>, CycleError> {
        Ok([
            left.into_iter().map(add_node).collect(),
            right.into_iter().map(add_node).collect(),
        ])
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

    /// LiveClone returns a deep clone — mutating the clone does not affect
    /// the live graph.
    #[tokio::test]
    async fn live_clone_is_independent_of_source() {
        let live: BothEyes<Arc<RwLock<GraphMem<VectorId>>>> = [
            Arc::new(RwLock::new(GraphMem::new())),
            Arc::new(RwLock::new(GraphMem::new())),
        ];
        // Seed both eyes with a node so the clone has something to lose.
        for eye in &live {
            let mut g = eye.write().await;
            g.insert_apply(vec![GraphMutation::AddNode {
                id: vid(1),
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            }]);
        }

        let mut m = LiveClone::new([Arc::clone(&live[0]), Arc::clone(&live[1])]);
        let snap = m.snapshot(cp_meta(), FreezeHeight(7)).await.unwrap();
        assert_eq!(snap.actual_height, 7);

        // Drop nodes from the clone via in-place mutation — live graphs unchanged.
        let mut snap = snap;
        snap.graph[0].insert_apply(vec![GraphMutation::RemoveNode { id: vid(1) }]);
        let live_left = live[0].read().await;
        assert!(
            live_left
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(1)).is_some()),
            "live graph must still contain vid(1) after clone-side removal",
        );
    }

    /// `apply_wal_stream` applies each row's left and right mutations to the
    /// matching eye graph and preserves cross-row order.
    #[tokio::test]
    async fn apply_wal_stream_routes_to_correct_eye() {
        let mut graph: Graph = [GraphMem::new(), GraphMem::new()];

        // Three rows: left gains 10, right gains 20, then left gains 11 + right gains 21.
        let items = vec![
            row(vec![10], vec![]),
            row(vec![], vec![20]),
            row(vec![11], vec![21]),
        ];

        apply_wal_stream(&mut graph, stream::iter(items))
            .await
            .unwrap();

        let left_has = |n| {
            graph[LEFT]
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(n)).is_some())
        };
        let right_has = |n| {
            graph[RIGHT]
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(n)).is_some())
        };

        assert!(left_has(10));
        assert!(left_has(11));
        assert!(!left_has(20), "20 must not leak into LEFT");
        assert!(!left_has(21), "21 must not leak into LEFT");

        assert!(right_has(20));
        assert!(right_has(21));
        assert!(!right_has(10), "10 must not leak into RIGHT");
        assert!(!right_has(11), "11 must not leak into RIGHT");
    }

    /// `apply_wal_stream` propagates a CycleError from any row in the stream.
    #[tokio::test]
    async fn apply_wal_stream_propagates_stream_error() {
        let mut graph: Graph = [GraphMem::new(), GraphMem::new()];
        let items = vec![
            row(vec![1], vec![2]),
            Err(CycleError::Fatal("boom".into())),
            row(vec![3], vec![4]),
        ];

        let err = apply_wal_stream(&mut graph, stream::iter(items))
            .await
            .unwrap_err();
        assert!(matches!(err, CycleError::Fatal(_)));

        // Mutations from rows before the error are applied; after the error are not.
        let left_has = |n| {
            graph[LEFT]
                .get_layers()
                .iter()
                .any(|l| l.get_links(&vid(n)).is_some())
        };
        assert!(left_has(1), "row before the error should have applied");
        assert!(!left_has(3), "row after the error should not have applied");
    }
}
