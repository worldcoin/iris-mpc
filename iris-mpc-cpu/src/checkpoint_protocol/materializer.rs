use async_trait::async_trait;
use aws_sdk_s3::Client as S3Client;

use crate::checkpoint_protocol::{
    CheckpointMeta, CycleError, FreezeHeight, Graph, Materializer, MutationStore,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::{
    download_graph, s3_key_exists, stream_download_and_deserialize_graph_pair,
};
use crate::hnsw::{graph::graph_store::GraphPg, VectorStore};
use crate::utils::serialization::graph::{read_graph_pair, GraphFormat};
use futures::TryStreamExt;

/// How the base checkpoint bytes are fetched during materialization.
#[derive(Clone, Copy, Debug)]
pub enum CheckpointDownload {
    /// Stream ranged GETs through the decoder; peak memory is bounded to the
    /// download window plus the graph. Used by the sidecar.
    Streaming,
    /// Buffer the whole object, then deserialize from memory. Higher peak RSS
    /// (full file resident alongside the graph); the established hawk-restart path.
    Buffered,
}

/// Rebuilds the graph from an S3 checkpoint plus WAL replay.
pub struct RebuildFromCheckpoint<'a, V: VectorStore> {
    pub graph_store: &'a GraphPg<V>,
    pub s3_client: &'a S3Client,
    pub bucket: String,
    pub download: CheckpointDownload,
}

impl<'a, V: VectorStore + Send + Sync> RebuildFromCheckpoint<'a, V> {
    pub fn new(
        graph_store: &'a GraphPg<V>,
        s3_client: &'a S3Client,
        bucket: String,
        download: CheckpointDownload,
    ) -> Self {
        Self {
            graph_store,
            s3_client,
            bucket,
            download,
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
        // The materializer rebuilds without iris-store access, so it cannot prune
        // a legacy base; it requires a native V5 checkpoint. Migrating V3/V4 to
        // V5 is genesis's job (it prunes against the registry).
        let format = GraphFormat::try_from(base.graph_version)
            .ok()
            .filter(|f| matches!(f, GraphFormat::V5))
            .ok_or_else(|| {
                CycleError::Fatal(format!(
                    "materializer requires a V5 checkpoint (legacy bases are migrated by \
                     genesis); got graph_version={} for {}/{}",
                    base.graph_version, self.bucket, base.s3_key,
                ))
            })?;

        // Fetch + deserialize the base checkpoint. `Streaming` feeds parallel
        // ranged GETs through the decoder on a blocking thread, bounding peak
        // memory to the download window plus the graph; `Buffered` downloads the
        // whole object first (full file resident alongside the graph). Both
        // return BLAKE3 over the downloaded bytes for verification below.
        let label = format!("{}/{}", self.bucket, base.s3_key);
        let (mut graph, downloaded_hash) = match self.download {
            CheckpointDownload::Streaming => stream_download_and_deserialize_graph_pair(
                self.s3_client,
                &self.bucket,
                &base.s3_key,
                format,
                None,
            )
            .await
            .map_err(|e| {
                CycleError::Fatal(format!(
                    "stream_download_and_deserialize_graph_pair({label}): {e}"
                ))
            })?,
            CheckpointDownload::Buffered => {
                let bytes = download_graph(self.s3_client, &self.bucket, &base.s3_key)
                    .await
                    .map_err(|e| CycleError::Fatal(format!("download_graph({label}): {e}")))?;
                // hash + deserialize are CPU-bound; run off the async runtime.
                tokio::task::spawn_blocking(move || {
                    let hash = *blake3::hash(&bytes).as_bytes();
                    let graph = read_graph_pair(&mut std::io::Cursor::new(&bytes), format)
                        .map_err(|e| CycleError::Fatal(format!("read_graph_pair({label}): {e}")))?;
                    Ok::<_, CycleError>((graph, hash))
                })
                .await
                .map_err(|e| CycleError::Fatal(format!("materialize deserialize task: {e}")))??
            }
        };

        let downloaded_hex = hex::encode(downloaded_hash);
        if downloaded_hex != base.blake3_hash {
            // The row's hash does not describe the object it points at. The
            // row is tombstoned; the object is deliberately left in place:
            //
            //  * the tombstone is party-local and reversible (one UPDATE flips
            //    `is_deleted` back, and the row stays readable via
            //    `get_genesis_graph_checkpoints_including_deleted`), so doing
            //    it automatically forecloses nothing;
            //  * deleting the object is neither. It is the only copy of the
            //    evidence, and if the fault is a bad hash in the row rather
            //    than bad bytes in S3, those bytes are still the good graph.
            //
            // So this is not a repair: it stops a base that no party can
            // materialize from winning Phase 1 again, and nothing more.
            // Whether the object is corrupt, superseded or tampered with —
            // and whether it may be deleted — is an operator call.
            //
            // The tombstone also hides the row from the next cycle's check, so
            // the mismatch is detected exactly once and later cycles go green
            // on an older base. The counter and log below are therefore the
            // signal to alert on, not the returned error.
            metrics::counter!("checkpoint_base_hash_mismatch_total").increment(1);
            let soft_deleted = match self
                .graph_store
                .delete_genesis_checkpoint(base.checkpoint_id)
                .await
            {
                Ok(()) => {
                    tracing::error!(
                        checkpoint_id = base.checkpoint_id,
                        s3_key = %base.s3_key,
                        bucket = %self.bucket,
                        expected = %base.blake3_hash,
                        got = %downloaded_hex,
                        "BLAKE3 mismatch: checkpoint row soft-deleted, s3 object retained \
                         for inspection; operator must triage before it can be reused"
                    );
                    true
                }
                Err(e) => {
                    tracing::error!(
                        checkpoint_id = base.checkpoint_id,
                        error = %e,
                        "failed to soft-delete checkpoint after BLAKE3 mismatch"
                    );
                    false
                }
            };
            let detail = format!(
                "BLAKE3 mismatch for {}/{}: expected={} got={} (checkpoint_id={})",
                self.bucket, base.s3_key, base.blake3_hash, downloaded_hex, base.checkpoint_id,
            );
            // Retryable only because the tombstone landed: the next cycle
            // cannot agree on this row again, so it will converge on an older
            // base. If the tombstone failed the row is still visible, a retry
            // would re-agree on it and spin — that case has to stop the caller.
            return Err(if soft_deleted {
                CycleError::BaseRejected(format!("{detail}; row soft-deleted, retry will re-base"))
            } else {
                CycleError::Fatal(format!("{detail}; row could NOT be soft-deleted"))
            });
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

    async fn filter_available(
        &self,
        candidates: Vec<CheckpointMeta>,
    ) -> Result<Vec<CheckpointMeta>, CycleError> {
        let mut available = Vec::with_capacity(candidates.len());
        for meta in candidates {
            // Anything other than a clean "absent" answer is transient: the
            // caller retries rather than discarding a base on a flaky probe.
            let exists = s3_key_exists(self.s3_client, &self.bucket, &meta.s3_key)
                .await
                .map_err(|e| {
                    CycleError::Transient(format!(
                        "s3_key_exists({}/{}): {e}",
                        self.bucket, meta.s3_key
                    ))
                })?;
            if exists {
                available.push(meta);
            } else {
                metrics::counter!("checkpoint_base_missing_object_total").increment(1);
                tracing::warn!(
                    checkpoint_id = meta.checkpoint_id,
                    s3_key = %meta.s3_key,
                    bucket = %self.bucket,
                    "checkpoint row has no object in s3; excluding it from base proposals"
                );
            }
        }
        Ok(available)
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
    use iris_mpc_common::VectorId;

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    // Use `n` itself as the seq_no — every test below picks node ids that
    // are strictly increasing within a given eye, so this satisfies the
    // strict-increase invariant `insert_apply_all` enforces.
    fn add_node(n: u32) -> GraphMutation {
        GraphMutation {
            seq_no: n as u64,
            as_of: (n as u64) - 1,
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
                .any(|l| l.get_links(&n).is_some())
        };
        let right_has = |n| {
            graph[RIGHT]
                .get_layers()
                .iter()
                .any(|l| l.get_links(&n).is_some())
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
                .any(|l| l.get_links(&n).is_some())
        };
        assert!(left_has(1), "row before the error should have applied");
        assert!(!left_has(3), "row after the error should not have applied");
    }
}
