use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;

use crate::checkpoint_protocol::{
    CheckpointMeta, CycleError, FreezeHeight, Graph, Materializer, MutationStore,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::stream_download_and_deserialize_graph_pair;
use crate::hnsw::{graph::graph_store::GraphPg, VectorStore};
use crate::utils::serialization::graph::GraphFormat;
use futures::TryStreamExt;
use iris_mpc_common::vector_id::VectorId;

/// Rebuilds the graph from an S3 checkpoint plus WAL replay.
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
        let format = GraphFormat::try_from(base.graph_version)
            .ok()
            .filter(|f| matches!(f, GraphFormat::V3 | GraphFormat::V4))
            .ok_or_else(|| {
                CycleError::Fatal(format!(
                    "unsupported checkpoint graph_version={} for {}/{}",
                    base.graph_version, self.bucket, base.s3_key,
                ))
            })?;

        let (mut graph, downloaded_hash) = stream_download_and_deserialize_graph_pair(
            self.s3_client,
            &self.bucket,
            &base.s3_key,
            format,
        )
        .await
        .map_err(|e| {
            CycleError::Fatal(format!(
                "stream_download_and_deserialize_graph_pair({}/{}): {e}",
                self.bucket, base.s3_key,
            ))
        })?;

        let downloaded_hex = hex::encode(downloaded_hash);
        if downloaded_hex != base.blake3_hash {
            return Err(CycleError::Fatal(format!(
                "BLAKE3 mismatch for {}/{}: expected={} got={}",
                self.bucket, base.s3_key, base.blake3_hash, downloaded_hex,
            )));
        }

        // Replay WAL rows in `(base.graph_mutation_id, freeze]`. The Runner's
        // `PeerBehindBase` skip ensures `freeze >= lo` here; `hi == lo` is a
        // valid empty replay.
        let lo = base.graph_mutation_id.unwrap_or(0);
        let hi = freeze.0;
        if hi < lo {
            return Err(CycleError::Fatal(format!(
                "materializer invariant violated: freeze ({hi}) < base.graph_mutation_id ({lo})"
            )));
        }
        tracing::info!(
            blake3 = %downloaded_hex,
            from = lo,
            to = hi,
            "materialize: base verified, replaying WAL range (from, to]"
        );
        let stream = MutationStore::mutations_in_range(self.graph_store, lo, hi).await?;
        let applied = apply_wal_stream(&mut graph, stream).await?;
        tracing::info!(rows = applied, "materialize: WAL replay complete");

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
        Result<BothEyes<Vec<crate::hnsw::graph::mutation::GraphMutation>>, CycleError>,
    >,
) -> Result<usize, CycleError> {
    use crate::execution::hawk_main::{LEFT, RIGHT};
    let mut applied = 0usize;
    while let Some(row) = stream.try_next().await? {
        let [left_muts, right_muts] = row;
        graph[LEFT]
            .insert_apply_all(&left_muts)
            .map_err(|e| CycleError::Fatal(format!("WAL replay (LEFT) failed: {e}")))?;
        graph[RIGHT]
            .insert_apply_all(&right_muts)
            .map_err(|e| CycleError::Fatal(format!("WAL replay (RIGHT) failed: {e}")))?;
        applied += 1;
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::hawk_main::{LEFT, RIGHT};
    use crate::hnsw::graph::layered_graph::GraphMem;
    use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
    use futures::{stream, StreamExt};

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    // Use `n` itself as the seq_no — every test below picks node ids that
    // are strictly increasing within a given eye, so this satisfies the
    // strict-increase invariant `insert_apply_all` enforces.
    fn add_node(n: u32) -> GraphMutation {
        GraphMutation {
            seq_no: n as u64,
            ops: vec![MutationOp::AddNode {
                id: vid(n),
                height: 1,
                update_ep: UpdateEntryPoint::False,
            }],
        }
    }

    fn row(left: Vec<u32>, right: Vec<u32>) -> Result<BothEyes<Vec<GraphMutation>>, CycleError> {
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
