//! Postgres-backed [`MutationStore`] for the checkpoint protocol.
//!
//! Adapts [`GraphPg`] to the trait surface that `run_cycle` consumes:
//! recent genesis checkpoints (newest first), a streaming WAL range over
//! `hawk_graph_mutations`, and the live max `modification_id`.

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt, TryStreamExt};

use crate::checkpoint_protocol::{CheckpointMeta, CycleError, GraphMutationId, MutationStore};
use crate::execution::hawk_main::BothEyes;
use crate::hnsw::{
    graph::{graph_store::GraphPg, mutation::GraphMutation},
    VectorStore,
};

#[async_trait]
impl<V: VectorStore + Send + Sync> MutationStore for GraphPg<V> {
    async fn recent_checkpoints(&self, window: usize) -> Result<Vec<CheckpointMeta>, CycleError> {
        let rows = self
            .get_genesis_graph_checkpoints()
            .await
            .map_err(|e| CycleError::Transient(format!("get_genesis_graph_checkpoints: {e}")))?;

        // `get_genesis_graph_checkpoints` returns DESC by id (newest first).
        Ok(rows
            .into_iter()
            .take(window)
            .map(|row| CheckpointMeta {
                checkpoint_id: row.id,
                s3_key: row.s3_key,
                last_indexed_iris_id: row.last_indexed_iris_id,
                last_indexed_modification_id: row.last_indexed_modification_id,
                graph_mutation_id: row.graph_mutation_id,
                blake3_hash: row.blake3_hash,
                graph_version: row.graph_version,
            })
            .collect())
    }

    async fn mutations_in_range(
        &self,
        lo_exclusive: GraphMutationId,
        hi_inclusive: GraphMutationId,
    ) -> Result<BoxStream<'_, Result<BothEyes<Vec<GraphMutation>>, CycleError>>, CycleError> {
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

    async fn current_max_mutation_id(&self) -> Result<GraphMutationId, CycleError> {
        let max = self
            .get_max_hawk_graph_mutation_id()
            .await
            .map_err(|e| CycleError::Fatal(format!("get_max_hawk_graph_mutation_id: {e}")))?;
        Ok(max.unwrap_or(0))
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use super::*;
    use crate::hawkers::plaintext_store::PlaintextStore;
    use crate::hnsw::graph::{
        graph_store::test_utils::TestGraphPg,
        mutation::{GraphMutation, MutationOp},
    };
    use futures::TryStreamExt;
    use iris_mpc_common::vector_id::VectorId;

    fn vid(n: u32) -> VectorId {
        VectorId::from_serial_id(n)
    }

    /// Builds a `BothEyes<Vec<GraphMutation>>` with one RemoveNode per eye,
    /// targeting different ids so left/right are distinguishable in the test.
    fn both_eyes_payload(left_id: u32, right_id: u32) -> BothEyes<Vec<GraphMutation>> {
        let mk = |id: u32| GraphMutation {
            seq_no: 1,
            ops: vec![MutationOp::RemoveNode { id: vid(id) }],
        };
        [vec![mk(left_id)], vec![mk(right_id)]]
    }

    async fn insert_row(
        store: &TestGraphPg<PlaintextStore>,
        modification_id: i64,
        payload: &BothEyes<Vec<GraphMutation>>,
    ) -> eyre::Result<()> {
        let bytes = bincode::serialize(payload)?;
        let mut graph_tx = store.tx().await?;
        store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, modification_id, &bytes)
            .await?;
        graph_tx.tx.commit().await?;
        Ok(())
    }

    /// `recent_checkpoints` is empty on an empty table; returns rows
    /// newest-first when populated.
    #[tokio::test]
    async fn test_recent_checkpoints_round_trip() -> eyre::Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let rows = MutationStore::recent_checkpoints(&store.graph, 10).await?;
        assert!(
            rows.is_empty(),
            "empty genesis_graph_checkpoint should give empty list"
        );

        // Insert two checkpoints; expect newest-first ordering.
        for (s3_key, iris_id, mut_id) in [("cp/42", 123_i64, 456_i64), ("cp/43", 200_i64, 500_i64)]
        {
            let mut graph_tx = store.tx().await?;
            GraphPg::<PlaintextStore>::insert_genesis_graph_checkpoint(
                &mut graph_tx.tx,
                s3_key,
                iris_id,
                mut_id,
                Some(mut_id + 1),
                "deadbeefcafebabe",
                false,
                1,
            )
            .await?;
            graph_tx.tx.commit().await?;
        }

        let rows = MutationStore::recent_checkpoints(&store.graph, 10).await?;
        assert_eq!(rows.len(), 2);
        // Newest first: cp/43 inserted second → id is higher.
        assert_eq!(rows[0].s3_key, "cp/43");
        assert_eq!(rows[1].s3_key, "cp/42");

        // `window` truncates.
        let rows = MutationStore::recent_checkpoints(&store.graph, 1).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].s3_key, "cp/43");

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
        let rows: Vec<BothEyes<Vec<GraphMutation>>> = stream.try_collect().await?;

        assert_eq!(rows.len(), 2, "(1,9] should include ids 5 and 9");
        for (row, expected) in rows.iter().zip([(105, 205), (109, 209)]) {
            let [left, right] = row;
            assert!(
                matches!(&left[0].ops[..], [MutationOp::RemoveNode { id }] if *id == vid(expected.0))
            );
            assert!(
                matches!(&right[0].ops[..], [MutationOp::RemoveNode { id }] if *id == vid(expected.1))
            );
        }

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
