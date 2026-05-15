//! Postgres-backed [`MutationStore`] for the checkpoint protocol.
//!
//! Adapts [`GraphPg`] to the trait surface that `run_cycle` consumes:
//! the latest genesis checkpoint, a streaming WAL range over
//! `hawk_graph_mutations`, the persisted `last_indexed_modification_id`,
//! and the live max `modification_id`.

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt, TryStreamExt};

use crate::checkpoint_protocol::{CheckpointMeta, CycleError, GraphMutationId, MutationStore};
use crate::execution::hawk_main::BothEyes;
use crate::hnsw::{
    graph::{graph_store::GraphPg, mutation::GraphMutation},
    VectorStore,
};
use iris_mpc_common::vector_id::VectorId;

const HAWK_DOMAIN: &str = "hawk";
const LAST_INDEXED_MODIFICATION_ID_KEY: &str = "last_indexed_modification_id";

fn transient<E: std::fmt::Display>(ctx: &str) -> impl FnOnce(E) -> CycleError + '_ {
    move |e| CycleError::Transient(format!("{ctx}: {e}"))
}

fn fatal<E: std::fmt::Display>(ctx: &str) -> impl FnOnce(E) -> CycleError + '_ {
    move |e| CycleError::Fatal(format!("{ctx}: {e}"))
}

#[async_trait]
impl<V: VectorStore + Send + Sync> MutationStore for GraphPg<V> {
    async fn latest_checkpoint(&self) -> Result<CheckpointMeta, CycleError> {
        let row = self
            .get_latest_genesis_graph_checkpoint()
            .await
            .map_err(transient("get_latest_genesis_graph_checkpoint"))?
            .ok_or_else(|| {
                CycleError::Fatal(
                    "no genesis_graph_checkpoint row exists; cannot start a cycle".into(),
                )
            })?;

        Ok(CheckpointMeta {
            checkpoint_id: row.id,
            s3_key: row.s3_key,
            last_indexed_iris_id: row.last_indexed_iris_id,
            last_indexed_modification_id: row.last_indexed_modification_id,
            graph_mutation_id: row.graph_mutation_id,
            blake3_hash: row.blake3_hash,
        })
    }

    async fn mutations_in_range(
        &self,
        lo_exclusive: GraphMutationId,
        hi_inclusive: GraphMutationId,
    ) -> Result<BoxStream<'_, Result<BothEyes<Vec<GraphMutation<VectorId>>>, CycleError>>, CycleError>
    {
        let stream = self
            .stream_hawk_graph_mutations_in_range(lo_exclusive, hi_inclusive)
            .map_err(|e| CycleError::Transient(format!("stream hawk_graph_mutations: {e}")))
            .and_then(|row| async move {
                row.deserialize_mutations().map_err(|e| {
                    CycleError::Fatal(format!(
                        "deserialize mutations at id={}: {e}",
                        row.modification_id,
                    ))
                })
            });
        Ok(stream.boxed())
    }

    async fn last_indexed_modification_id(&self) -> Result<i64, CycleError> {
        let value: Option<i64> = self
            .get_persistent_state(HAWK_DOMAIN, LAST_INDEXED_MODIFICATION_ID_KEY)
            .await
            .map_err(transient(
                "get_persistent_state(last_indexed_modification_id)",
            ))?;
        Ok(value.unwrap_or(0))
    }

    async fn current_max_mutation_id(&self) -> Result<GraphMutationId, CycleError> {
        let max = self
            .get_max_hawk_graph_mutation_id()
            .await
            .map_err(fatal("get_max_hawk_graph_mutation_id"))?;
        Ok(max.unwrap_or(0))
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use super::*;
    use crate::hawkers::plaintext_store::PlaintextStore;
    use crate::hnsw::graph::{graph_store::test_utils::TestGraphPg, mutation::GraphMutation};
    use futures::TryStreamExt;
    use iris_mpc_common::vector_id::VectorId;

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    /// Builds a `BothEyes<Vec<GraphMutation<VectorId>>>` with one RemoveNode per eye,
    /// targeting different ids so left/right are distinguishable in the test.
    fn both_eyes_payload(left_id: u32, right_id: u32) -> BothEyes<Vec<GraphMutation<VectorId>>> {
        [
            vec![GraphMutation::RemoveNode { id: vid(left_id) }],
            vec![GraphMutation::RemoveNode { id: vid(right_id) }],
        ]
    }

    async fn insert_row(
        store: &TestGraphPg<PlaintextStore>,
        modification_id: i64,
        payload: &BothEyes<Vec<GraphMutation<VectorId>>>,
    ) -> eyre::Result<()> {
        let bytes = bincode::serialize(payload)?;
        let mut graph_tx = store.tx().await?;
        store
            .insert_hawk_graph_mutations(&mut graph_tx.tx, modification_id, &bytes)
            .await?;
        graph_tx.tx.commit().await?;
        Ok(())
    }

    /// `latest_checkpoint` errors when no genesis_graph_checkpoint row exists;
    /// returns the row's fields when one does.
    #[tokio::test]
    async fn test_latest_checkpoint_round_trip() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        // Empty table => Fatal (cycle cannot proceed without a base).
        let err = MutationStore::latest_checkpoint(&store.graph)
            .await
            .unwrap_err();
        assert!(
            matches!(err, CycleError::Fatal(_)),
            "expected Fatal, got {err:?}"
        );

        // Insert a checkpoint and read it back through the trait.
        let mut graph_tx = store.tx().await?;
        GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
            &mut graph_tx.tx,
            "cp/42",
            123,
            456,
            Some(789),
            "deadbeefcafebabe",
            false,
            1,
        )
        .await?;
        graph_tx.tx.commit().await?;

        let meta = MutationStore::latest_checkpoint(&store.graph).await?;
        assert_eq!(meta.s3_key, "cp/42");
        assert_eq!(meta.last_indexed_iris_id, 123);
        assert_eq!(meta.last_indexed_modification_id, 456);
        assert_eq!(meta.graph_mutation_id, Some(789));
        assert_eq!(meta.blake3_hash, "deadbeefcafebabe");

        Ok(())
    }

    /// `mutations_in_range` deserializes each row into `BothEyes<Vec<GraphMutation>>`,
    /// preserves left/right attribution, and yields rows in modification_id order.
    #[tokio::test]
    async fn test_mutations_in_range_round_trip() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        insert_row(&store, 1, &both_eyes_payload(101, 201)).await?;
        insert_row(&store, 5, &both_eyes_payload(105, 205)).await?;
        insert_row(&store, 9, &both_eyes_payload(109, 209)).await?;

        let stream = MutationStore::mutations_in_range(&store.graph, 1, 9).await?;
        let rows: Vec<BothEyes<Vec<GraphMutation<VectorId>>>> = stream.try_collect().await?;

        assert_eq!(rows.len(), 2, "(1,9] should include ids 5 and 9");
        for (row, expected) in rows.iter().zip([(105, 205), (109, 209)]) {
            let [left, right] = row;
            assert!(matches!(left[0], GraphMutation::RemoveNode { id } if id == vid(expected.0)));
            assert!(matches!(right[0], GraphMutation::RemoveNode { id } if id == vid(expected.1)));
        }

        Ok(())
    }

    /// `last_indexed_modification_id` defaults to 0 when the row is missing.
    #[tokio::test]
    async fn test_last_indexed_modification_id_defaults_to_zero() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;
        let v = MutationStore::last_indexed_modification_id(&store.graph).await?;
        assert_eq!(v, 0);
        Ok(())
    }

    /// `last_indexed_modification_id` reads back whatever was written via set_persistent_state.
    #[tokio::test]
    async fn test_last_indexed_modification_id_reads_persistent_state() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let mut graph_tx = store.tx().await?;
        GraphPg::<PlaintextStore>::set_persistent_state(
            &mut graph_tx.tx,
            HAWK_DOMAIN,
            LAST_INDEXED_MODIFICATION_ID_KEY,
            &123i64,
        )
        .await?;
        graph_tx.tx.commit().await?;

        let v = MutationStore::last_indexed_modification_id(&store.graph).await?;
        assert_eq!(v, 123);
        Ok(())
    }

    /// `current_max_mutation_id` is 0 on an empty table and tracks inserts.
    #[tokio::test]
    async fn test_current_max_mutation_id() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        assert_eq!(
            MutationStore::current_max_mutation_id(&store.graph).await?,
            0
        );

        insert_row(&store, 7, &both_eyes_payload(1, 2)).await?;
        assert_eq!(
            MutationStore::current_max_mutation_id(&store.graph).await?,
            7
        );

        Ok(())
    }
}
