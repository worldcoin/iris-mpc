use crate::{
    execution::hawk_main::BothEyes,
    hnsw::{graph::GraphMutation, VectorStore},
    utils::serialization::graph_mutation::{deserialize_mutations, GraphMutationFormat},
};
use eyre::{eyre, Result};
use iris_mpc_common::postgres::PostgresClient;
use serde::{de::DeserializeOwned, Serialize};
use sqlx::{types::Json, Postgres, Row, Transaction};
use std::{marker::PhantomData, ops::DerefMut};

#[derive(sqlx::FromRow, Debug, Clone, PartialEq, Eq)]
pub struct GraphCheckpointRow {
    pub id: i64,
    pub s3_key: String,
    pub last_indexed_iris_id: i64,
    pub last_indexed_modification_id: i64,
    pub graph_mutation_id: Option<i64>,
    pub blake3_hash: String,
    pub graph_version: i32,
    pub is_archival: bool,
}

/// A row from the hawk_graph_mutations table.
#[derive(sqlx::FromRow, Debug, Clone, PartialEq, Eq)]
pub struct GraphMutationRow {
    pub modification_id: i64,
    /// Bincode-serialized `BothEyes<Vec<GraphMutation>>` (mutations for both eyes)
    pub serialized_mutations: Vec<u8>,
    pub mutation_format_version: i16,
}

pub struct GraphPg<V: VectorStore> {
    pool: sqlx::PgPool,
    schema_name: String,
    phantom: PhantomData<V>,
}

impl GraphMutationRow {
    pub fn deserialize_mutations(&self) -> Result<BothEyes<Vec<GraphMutation>>> {
        let format = GraphMutationFormat::try_from(self.mutation_format_version)?;
        deserialize_mutations(format, &self.serialized_mutations)
    }
}

impl<V: VectorStore> GraphPg<V> {
    pub async fn new(postgres_client: &PostgresClient) -> Result<Self> {
        tracing::info!(
            "Created a graph store with schema: {}",
            postgres_client.schema_name,
        );

        Ok(Self {
            pool: postgres_client.pool.clone(),
            schema_name: postgres_client.schema_name.clone(),
            phantom: PhantomData,
        })
    }

    pub fn pool(&self) -> &sqlx::PgPool {
        &self.pool
    }

    pub async fn tx(&self) -> Result<GraphTx<'_, V>> {
        Ok(self.tx_wrap(self.pool.begin().await?))
    }

    pub fn tx_wrap<'t>(&self, tx: Transaction<'t, Postgres>) -> GraphTx<'t, V> {
        GraphTx {
            tx,
            schema_name: self.schema_name.clone(),
            phantom: PhantomData,
        }
    }

    /// Retrieve entry from `persistent_state` table with associated `domain` and `key` identifiers.
    ///
    /// Returns `Some(value)` if an entry exists for these identifiers and deserialization to generic
    /// type `T` succeeds.  If no entry exists, then returns `None`.
    pub async fn get_persistent_state<T: DeserializeOwned>(
        &self,
        domain: &str,
        key: &str,
    ) -> Result<Option<T>> {
        let row = sqlx::query(
            r#"
            SELECT "value"
            FROM persistent_state
            WHERE domain = $1 AND "key" = $2
            "#,
        )
        .bind(domain)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;

        if let Some(row) = row {
            let value = row.try_get("value")?;
            let deserialized: T = serde_json::from_value(value)?;
            Ok(Some(deserialized))
        } else {
            Ok(None)
        }
    }

    /// Set the entry in the `persistent_state` table for primary key `(domain, key)`.
    ///
    /// If an entry already exists for this primary key, this overwrites the existing
    /// value with the value specified here.
    pub async fn set_persistent_state<T: Serialize>(
        tx: &mut Transaction<'_, Postgres>,
        domain: &str,
        key: &str,
        value: &T,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO persistent_state (domain, "key", "value")
            VALUES ($1, $2, $3)
            ON CONFLICT (domain, "key")
            DO UPDATE SET "value" = EXCLUDED."value"
            "#,
        )
        .bind(domain)
        .bind(key)
        .bind(Json(value))
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Delete an entry in the `persistent_state` table with primary key `(domain, key)`,
    /// if it exists.
    pub async fn delete_persistent_state(
        tx: &mut Transaction<'_, Postgres>,
        domain: &str,
        key: &str,
    ) -> Result<()> {
        // let value_json = Json(value);

        sqlx::query(
            r#"
            DELETE FROM persistent_state
            WHERE domain = $1 AND "key" = $2;
            "#,
        )
        .bind(domain)
        .bind(key)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn insert_genesis_graph_checkpoint(
        tx: &mut Transaction<'_, Postgres>,
        s3_key: &str,
        last_indexed_iris_id: i64,
        last_indexed_modification_id: i64,
        graph_mutation_id: Option<i64>,
        blake3_hash: &str,
        is_archival: bool,
        graph_version: i32,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO genesis_graph_checkpoint (
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_mutation_id,
                blake3_hash,
                is_archival,
                graph_version
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
        )
        .bind(s3_key)
        .bind(last_indexed_iris_id)
        .bind(last_indexed_modification_id)
        .bind(graph_mutation_id)
        .bind(blake3_hash)
        .bind(is_archival)
        .bind(graph_version)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Returns the most recent genesis graph checkpoint
    pub async fn get_latest_genesis_graph_checkpoint(&self) -> Result<Option<GraphCheckpointRow>> {
        let row = sqlx::query_as::<_, GraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_mutation_id,
                blake3_hash,
                is_archival,
                graph_version
            FROM genesis_graph_checkpoint
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Returns a genesis graph checkpoint by its S3 key.
    pub async fn get_genesis_graph_checkpoint_by_key(
        &self,
        s3_key: &str,
    ) -> Result<Option<GraphCheckpointRow>> {
        let row = sqlx::query_as::<_, GraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_mutation_id,
                blake3_hash,
                is_archival,
                graph_version
            FROM genesis_graph_checkpoint
            WHERE s3_key = $1
            "#,
        )
        .bind(s3_key)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Returns the newest genesis graph checkpoint with the given blake3 hash,
    /// from anywhere in checkpoint history.
    pub async fn get_genesis_graph_checkpoint_by_hash(
        &self,
        blake3_hash: &str,
    ) -> Result<Option<GraphCheckpointRow>> {
        let row = sqlx::query_as::<_, GraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_mutation_id,
                blake3_hash,
                is_archival,
                graph_version
            FROM genesis_graph_checkpoint
            WHERE blake3_hash = $1
            ORDER BY id DESC
            LIMIT 1
            "#,
        )
        .bind(blake3_hash)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Returns genesis graph checkpoints in descending order
    pub async fn get_genesis_graph_checkpoints(&self) -> Result<Vec<GraphCheckpointRow>> {
        let rows = sqlx::query_as::<_, GraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_mutation_id,
                blake3_hash,
                is_archival,
                graph_version
            FROM genesis_graph_checkpoint
            ORDER BY id DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| eyre!("Failed to fetch genesis checkpoints: {e}"))?;
        Ok(rows)
    }

    pub async fn delete_genesis_checkpoint(&self, id: i64) -> Result<()> {
        let _ = sqlx::query(
            r#"
            DELETE FROM genesis_graph_checkpoint WHERE id = $1
            "#,
        )
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| eyre!("Failed to delete genesis checkpoint: {e}"))?;

        Ok(())
    }

    /// INVARIANT: this function is only given GraphMutaionFormat::Current mutations.
    /// Do not serialize into an old mutation format.
    pub async fn upsert_hawk_graph_mutations(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        modification_id: i64,
        serialized_mutations: &[u8],
    ) -> Result<Vec<GraphMutationRow>> {
        let rows = sqlx::query_as::<_, GraphMutationRow>(
            r#"
            INSERT INTO hawk_graph_mutations (modification_id, serialized_mutations, mutation_format_version)
            VALUES ($1, $2, $3)
            ON CONFLICT (modification_id) DO UPDATE
            SET serialized_mutations = EXCLUDED.serialized_mutations,
                mutation_format_version = EXCLUDED.mutation_format_version
            RETURNING modification_id, mutation_format_version, serialized_mutations
            "#,
        )
        .bind(modification_id)
        .bind(serialized_mutations)
        .bind(GraphMutationFormat::CURRENT.version())
        .fetch_all(tx.deref_mut())
        .await?;

        Ok(rows.into_iter().collect())
    }

    pub async fn get_hawk_graph_mutations(
        &self,
        max_graph_mutation_id: Option<i64>,
    ) -> Result<Vec<GraphMutationRow>> {
        let rows = sqlx::query_as::<_, GraphMutationRow>(
            r#"
            SELECT modification_id, mutation_format_version, serialized_mutations
            FROM hawk_graph_mutations
            WHERE $1::bigint IS NULL OR modification_id <= $1
            ORDER BY modification_id ASC
            "#,
        )
        .bind(max_graph_mutation_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().collect())
    }

    pub async fn get_hawk_graph_mutations_after(
        &self,
        min_id: Option<i64>,
    ) -> Result<Vec<GraphMutationRow>> {
        let rows = sqlx::query_as::<_, GraphMutationRow>(
            r#"
            SELECT modification_id, mutation_format_version, serialized_mutations
            FROM hawk_graph_mutations
            WHERE $1::bigint IS NULL OR modification_id > $1
            ORDER BY modification_id ASC
            "#,
        )
        .bind(min_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| eyre!("Failed to fetch mutations after {min_id:?}: {e}"))?;

        Ok(rows.into_iter().collect())
    }

    /// Streams hawk_graph_mutations rows with `lo_exclusive < modification_id <= hi_inclusive`
    /// in ascending order. Yields one row at a time so callers can apply mutations
    /// without buffering the full range — important for WAL replay over long gaps.
    ///
    /// **Snapshot caveat**: rows are streamed from the pool's connection,
    /// not from a `REPEATABLE READ` transaction. Replay determinism relies
    /// on inserts into `hawk_graph_mutations` being strictly monotone in
    /// `modification_id` and committed before any future row at a higher
    /// id becomes visible (true today via the write path). If that ever
    /// changes — back-fills, gap-filling inserts, late commits — wrap this
    /// call in a transaction with `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ`.
    pub fn stream_hawk_graph_mutations_in_range(
        &self,
        lo_exclusive: i64,
        hi_inclusive: i64,
    ) -> futures::stream::BoxStream<'_, Result<GraphMutationRow, sqlx::Error>> {
        use futures::StreamExt;
        sqlx::query_as::<_, GraphMutationRow>(
            r#"
            SELECT modification_id, mutation_format_version, serialized_mutations
            FROM hawk_graph_mutations
            WHERE modification_id > $1 AND modification_id <= $2
            ORDER BY modification_id ASC
            "#,
        )
        .bind(lo_exclusive)
        .bind(hi_inclusive)
        .fetch(&self.pool)
        .boxed()
    }

    /// Returns every row whose `modification_id >= from_id`, ordered ascending.
    /// Use this when you know the exact first modification you need (inclusive),
    /// in contrast to `get_hawk_graph_mutations_after()` which is exclusive.
    pub async fn get_hawk_graph_mutations_from(
        &self,
        from_id: i64,
    ) -> Result<Vec<GraphMutationRow>> {
        let rows = sqlx::query_as::<_, GraphMutationRow>(
            r#"
            SELECT modification_id, mutation_format_version, serialized_mutations
            FROM hawk_graph_mutations
            WHERE modification_id >= $1
            ORDER BY modification_id ASC
            "#,
        )
        .bind(from_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| eyre!("Failed to fetch mutations from {from_id}: {e}"))?;

        Ok(rows.into_iter().collect())
    }

    /// Opens a new database transaction on the underlying pool.
    pub async fn begin_tx(&self) -> Result<Transaction<'_, Postgres>> {
        self.pool
            .begin()
            .await
            .map_err(|e| eyre!("Failed to begin graph transaction: {e}"))
    }

    pub async fn delete_hawk_graph_mutations(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        max_mutation_id: Option<i64>,
    ) -> Result<()> {
        match max_mutation_id {
            Some(max_mutation_id) => {
                sqlx::query(
                    r#"
                    DELETE FROM hawk_graph_mutations
                    WHERE modification_id <= $1
                    "#,
                )
                .bind(max_mutation_id)
                .execute(tx.deref_mut())
                .await?;
            }
            None => {
                sqlx::query("DELETE FROM hawk_graph_mutations")
                    .execute(tx.deref_mut())
                    .await?;
            }
        }

        Ok(())
    }

    /// Returns the maximum modification_id from hawk_graph_mutations table, or None if empty.
    pub async fn get_max_hawk_graph_mutation_id(&self) -> Result<Option<i64>> {
        let row = sqlx::query(
            r#"
            SELECT MAX(modification_id) as max_id
            FROM hawk_graph_mutations
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(row.try_get("max_id")?)
    }
}

pub struct GraphTx<'a, V> {
    pub tx: Transaction<'a, Postgres>,
    #[allow(dead_code)]
    schema_name: String,
    phantom: PhantomData<V>,
}

impl<'b, V: VectorStore> GraphTx<'b, V> {
    /// Insert a single graph mutation row into hawk_graph_mutations.
    pub async fn upsert_hawk_graph_mutations(
        &mut self,
        modification_id: i64,
        serialized_mutations: &[u8],
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO hawk_graph_mutations (modification_id, serialized_mutations, mutation_format_version)
            VALUES ($1, $2, $3)
            ON CONFLICT (modification_id) DO UPDATE
            SET serialized_mutations = EXCLUDED.serialized_mutations,
                mutation_format_version = EXCLUDED.mutation_format_version
            "#,
        )
        .bind(modification_id)
        .bind(serialized_mutations)
        .bind(GraphMutationFormat::CURRENT.version())
        .execute(self.tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Update the last_indexed_modification_id in persistent_state.
    /// This tracks the highest modification_id that has been persisted to hawk_graph_mutations.
    pub async fn set_last_indexed_modification_id(&mut self, modification_id: i64) -> Result<()> {
        const HAWK_DOMAIN: &str = "hawk";
        const LAST_INDEXED_MODIFICATION_ID_KEY: &str = "last_indexed_modification_id";

        sqlx::query(
            r#"
            INSERT INTO persistent_state (domain, "key", "value")
            VALUES ($1, $2, $3)
            ON CONFLICT (domain, "key")
            DO UPDATE SET "value" = EXCLUDED."value"
            "#,
        )
        .bind(HAWK_DOMAIN)
        .bind(LAST_INDEXED_MODIFICATION_ID_KEY)
        .bind(Json(modification_id))
        .execute(self.tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Delete every row from `hawk_graph_mutations`. Used when resetting the
    /// HNSW schema to a checkpoint: the WAL must not carry mutations that
    /// post-date the base.
    pub async fn clear_hawk_graph_mutations(&mut self) -> Result<()> {
        sqlx::query("DELETE FROM hawk_graph_mutations")
            .execute(self.tx.deref_mut())
            .await?;
        Ok(())
    }

    /// Delete `genesis_graph_checkpoint` rows created after the pinned row
    /// (row id order == creation order). Abandoned-lineage entries left by
    /// prior runs — including same-height ones — must not win the next run's
    /// latest-common selection.
    pub async fn delete_checkpoints_after_id(&mut self, checkpoint_row_id: i64) -> Result<()> {
        sqlx::query("DELETE FROM genesis_graph_checkpoint WHERE id > $1")
            .bind(checkpoint_row_id)
            .execute(self.tx.deref_mut())
            .await?;
        Ok(())
    }
}

pub mod test_utils {
    use super::*;
    use iris_mpc_common::postgres::{run_migrations, AccessMode, PostgresClient};
    use iris_mpc_store::test_utils::{cleanup, temporary_name, test_db_url};
    use std::ops::{Deref, DerefMut};

    /// A test database. It creates a unique schema for each test. Call
    /// `cleanup` at the end of the test.
    ///
    /// Access the database with `&graph` or `graph.owned()`.
    pub struct TestGraphPg<V: VectorStore> {
        postgres_client: PostgresClient,
        pub graph: GraphPg<V>,
    }

    impl<V: VectorStore> TestGraphPg<V> {
        pub async fn new() -> Result<Self> {
            let schema_name = temporary_name();
            let postgres_client =
                PostgresClient::new(&test_db_url()?, &schema_name, AccessMode::ReadWrite).await?;
            run_migrations(&postgres_client.pool, false).await?;
            let graph = GraphPg::new(&postgres_client).await?;
            Ok(TestGraphPg {
                postgres_client,
                graph,
            })
        }

        pub async fn cleanup(&self) -> Result<()> {
            cleanup(&self.postgres_client, &self.postgres_client.schema_name).await
        }

        pub fn owned(&self) -> GraphPg<V> {
            GraphPg {
                pool: self.graph.pool.clone(),
                schema_name: self.graph.schema_name.clone(),
                phantom: PhantomData,
            }
        }
    }

    impl<V: VectorStore> Deref for TestGraphPg<V> {
        type Target = GraphPg<V>;
        fn deref(&self) -> &Self::Target {
            &self.graph
        }
    }

    impl<V: VectorStore> DerefMut for TestGraphPg<V> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.graph
        }
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use super::{test_utils::TestGraphPg, *};
    use crate::hawkers::plaintext_store::PlaintextStore;
    use iris_mpc_common::VectorId;
    use tokio;

    #[tokio::test]
    async fn test_persistent_state() -> Result<()> {
        // Set up a temporary schema and a new store.
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let domain = "test_domain".to_string();
        let key = "foo".to_string();

        // Check no value at index
        let value: Option<String> = store.get_persistent_state(&domain, &key).await?;
        assert_eq!(value, None);

        // Insert value at index
        let set_value = "bar".to_string();
        let graph_tx = store.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::set_persistent_state(&mut tx, &domain, &key, &set_value).await?;
        tx.commit().await?;

        // Check value is set at index
        let value: Option<String> = store.get_persistent_state(&domain, &key).await?;
        assert_eq!(value, Some(set_value));

        // Insert new value at index
        let new_set_value = "bear".to_string();
        let graph_tx = store.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::set_persistent_state(&mut tx, &domain, &key, &new_set_value)
            .await?;
        tx.commit().await?;

        // Check value is updated at index
        let value: Option<String> = store.get_persistent_state(&domain, &key).await?;
        assert_eq!(value, Some(new_set_value));

        // Delete value at index
        let graph_tx = store.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::delete_persistent_state(&mut tx, &domain, &key).await?;
        tx.commit().await?;

        // Check no value at index
        let value: Option<String> = store.get_persistent_state(&domain, &key).await?;
        assert_eq!(value, None);

        // Delete value at index again
        let graph_tx = store.tx().await?;
        let mut tx = graph_tx.tx;
        GraphPg::<PlaintextStore>::delete_persistent_state(&mut tx, &domain, &key).await?;
        tx.commit().await?;

        // Check still no value at index
        let value: Option<String> = store.get_persistent_state(&domain, &key).await?;
        assert_eq!(value, None);

        Ok(())
    }

    /// Round-trip test for `GraphPg::insert_hawk_graph_mutations`:
    /// verifies the RETURNING clause echoes back all three columns correctly.
    #[tokio::test]
    async fn test_insert_hawk_graph_mutations_round_trip() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let modification_id: i64 = 42;
        let payload: &[u8] = b"round_trip_payload";

        let mut graph_tx = store.tx().await?;
        let returned = store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, modification_id, payload)
            .await?;
        graph_tx.tx.commit().await?;

        assert_eq!(returned.len(), 1, "RETURNING should yield exactly one row");
        let row = &returned[0];
        assert_eq!(row.modification_id, modification_id);
        assert_eq!(row.serialized_mutations, payload);

        Ok(())
    }

    /// Tests `get_hawk_graph_mutations_after` for:
    ///   - empty table with None
    ///   - None after inserts (returns all rows in ASC order)
    ///   - Some(id) with rows above the threshold
    ///   - Some(id) equal to the highest id (returns nothing)
    ///   - Some(id) above all rows (returns nothing)
    #[tokio::test]
    async fn test_get_hawk_graph_mutations_after() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        // Empty table: None should return an empty vec
        let rows = store.get_hawk_graph_mutations_after(None).await?;
        assert!(rows.is_empty(), "expected empty vec on empty table");

        // Populate three rows with distinct modification_ids
        for &id in &[10i64, 20, 30] {
            let mut graph_tx = store.tx().await?;
            store
                .upsert_hawk_graph_mutations(&mut graph_tx.tx, id, &id.to_le_bytes())
                .await?;
            graph_tx.tx.commit().await?;
        }

        // None => all rows, ordered ASC by modification_id
        let rows = store.get_hawk_graph_mutations_after(None).await?;
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].modification_id, 10);
        assert_eq!(rows[1].modification_id, 20);
        assert_eq!(rows[2].modification_id, 30);

        // Some(10) => strictly greater, so [20, 30]
        let rows = store.get_hawk_graph_mutations_after(Some(10)).await?;
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].modification_id, 20);
        assert_eq!(rows[1].modification_id, 30);

        // Some(29) => only [30]
        let rows = store.get_hawk_graph_mutations_after(Some(29)).await?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].modification_id, 30);

        // Some(30) => threshold equal to max, nothing above it
        let rows = store.get_hawk_graph_mutations_after(Some(30)).await?;
        assert!(rows.is_empty(), "threshold == max should return empty");

        // Some(100) => threshold above all rows
        let rows = store.get_hawk_graph_mutations_after(Some(100)).await?;
        assert!(
            rows.is_empty(),
            "threshold above all rows should return empty"
        );

        Ok(())
    }

    /// Tests `get_max_hawk_graph_mutation_id`:
    ///   - returns None on an empty table
    ///   - tracks the running maximum correctly as rows are inserted
    #[tokio::test]
    async fn test_get_max_hawk_graph_mutation_id() -> Result<()> {
        let store = TestGraphPg::<PlaintextStore>::new().await?;

        // Empty table => None
        assert_eq!(store.get_max_hawk_graph_mutation_id().await?, None);

        // Insert id=5 => max is Some(5)
        let mut graph_tx = store.tx().await?;
        store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, 5, b"a")
            .await?;
        graph_tx.tx.commit().await?;
        assert_eq!(store.get_max_hawk_graph_mutation_id().await?, Some(5));

        // Insert id=3 (below current max) => max stays Some(5)
        let mut graph_tx = store.tx().await?;
        store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, 3, b"b")
            .await?;
        graph_tx.tx.commit().await?;
        assert_eq!(store.get_max_hawk_graph_mutation_id().await?, Some(5));

        // Insert id=10 => max becomes Some(10)
        let mut graph_tx = store.tx().await?;
        store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, 10, b"c")
            .await?;
        graph_tx.tx.commit().await?;
        assert_eq!(store.get_max_hawk_graph_mutation_id().await?, Some(10));

        Ok(())
    }

    /// Tests `stream_hawk_graph_mutations_in_range`:
    ///   - empty table on any range
    ///   - half-open semantics: `(lo, hi]` excludes lo, includes hi
    ///   - empty when lo == hi
    ///   - rows below lo and above hi are excluded
    ///   - rows are streamed in ascending modification_id order
    #[tokio::test]
    async fn test_stream_hawk_graph_mutations_in_range() -> Result<()> {
        use futures::TryStreamExt;

        let store = TestGraphPg::<PlaintextStore>::new().await?;

        // Empty table => empty stream for any range.
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(0, 100)
            .try_collect()
            .await?;
        assert!(rows.is_empty());

        // Seed: ids 1, 5, 7, 10 (insert out of order to confirm ORDER BY ASC).
        let payloads: &[(i64, &[u8])] = &[(7, b"g"), (1, b"a"), (10, b"j"), (5, b"e")];
        for (id, p) in payloads {
            let mut graph_tx = store.tx().await?;
            store
                .upsert_hawk_graph_mutations(&mut graph_tx.tx, *id, p)
                .await?;
            graph_tx.tx.commit().await?;
        }

        // (0, 100]: all four, ascending.
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(0, 100)
            .try_collect()
            .await?;
        assert_eq!(
            rows.iter().map(|r| r.modification_id).collect::<Vec<_>>(),
            vec![1, 5, 7, 10]
        );

        // (1, 10]: excludes the lo bound, includes the hi bound.
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(1, 10)
            .try_collect()
            .await?;
        assert_eq!(
            rows.iter().map(|r| r.modification_id).collect::<Vec<_>>(),
            vec![5, 7, 10]
        );

        // (1, 7]: spans middle, excludes 10.
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(1, 7)
            .try_collect()
            .await?;
        assert_eq!(
            rows.iter().map(|r| r.modification_id).collect::<Vec<_>>(),
            vec![5, 7]
        );

        // (5, 5]: empty (degenerate range).
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(5, 5)
            .try_collect()
            .await?;
        assert!(rows.is_empty());

        // (50, 100]: empty (above all data).
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(50, 100)
            .try_collect()
            .await?;
        assert!(rows.is_empty());

        // Payload is round-tripped.
        let rows: Vec<GraphMutationRow> = store
            .stream_hawk_graph_mutations_in_range(0, 1)
            .try_collect()
            .await?;
        assert_eq!(rows[0].serialized_mutations, b"a");

        Ok(())
    }

    #[tokio::test]
    async fn test_hawk_graph_mutations_full_round_trip() -> Result<()> {
        use crate::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};

        let store = TestGraphPg::<PlaintextStore>::new().await?;

        let plan_left = GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![MutationOp::AddNode {
                id: VectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        };
        let plan_right = GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![MutationOp::AddNode {
                id: VectorId::from_serial_id(2),
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        };
        let both: BothEyes<Vec<GraphMutation>> =
            [vec![plan_left.clone()], vec![plan_right.clone()]];
        let payload = bincode::serialize(&both)?;

        let mut graph_tx = store.tx().await?;
        store
            .upsert_hawk_graph_mutations(&mut graph_tx.tx, 42, &payload)
            .await?;
        graph_tx.tx.commit().await?;

        let rows = store.get_hawk_graph_mutations_after(None).await?;
        assert_eq!(rows.len(), 1);
        // Full-record equality: this is the sole test covering the production
        // write path (bincode of the current type) read back through the V1
        // mirror, so it must catch any field drift between the two.
        let back = rows[0].deserialize_mutations()?;
        assert_eq!(back, both);

        store.cleanup().await?;
        Ok(())
    }
}
