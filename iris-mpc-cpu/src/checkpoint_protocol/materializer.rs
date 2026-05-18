//! [`Materializer`] implementation for the checkpoint protocol.
//!
//! [`RebuildFromCheckpoint`] downloads the base checkpoint from S3 and
//! replays WAL rows up to `freeze`. Deterministic; used by the sidecar
//! daemon and by Hawk restart. Both eyes of each WAL row are applied
//! atomically via [`GraphMem::insert_apply`], so left/right cannot drift.

use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;

use crate::checkpoint_protocol::{
    CheckpointMeta, CycleError, FreezeHeight, Graph, Materializer, MutationStore,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::{download_graph_checkpoint, GraphCheckpointState};
use crate::hnsw::{graph::graph_store::GraphPg, VectorStore};
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
    ) -> Result<Graph, CycleError> {
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

        Ok(graph)
    }
}

/// Applies a WAL stream to an in-memory graph. Each stream item is one
/// `hawk_graph_mutations` row's deserialized payload — both eyes together —
/// so the left-eye and right-eye mutations of a row are applied as a unit
/// before advancing.
async fn apply_wal_stream(
    graph: &mut Graph,
    mut stream: futures::stream::BoxStream<
        '_,
        Result<BothEyes<Vec<crate::hnsw::graph::mutation::GraphMutation<VectorId>>>, CycleError>,
    >,
) -> Result<(), CycleError> {
    use crate::execution::hawk_main::{LEFT, RIGHT};
    while let Some(row) = stream.try_next().await? {
        let [left_muts, right_muts] = row;
        graph[LEFT].insert_apply(left_muts);
        graph[RIGHT].insert_apply(right_muts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::hawk_main::{LEFT, RIGHT};
    use crate::hnsw::graph::layered_graph::GraphMem;
    use crate::hnsw::graph::mutation::{GraphMutation, UpdateEntryPoint};
    use futures::{stream, StreamExt};

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

        apply_wal_stream(&mut graph, stream::iter(items).boxed())
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

        let err = apply_wal_stream(&mut graph, stream::iter(items).boxed())
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
