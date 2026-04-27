use crate::{
    execution::hawk_main::StoreId,
    hnsw::{
        graph::{GraphMutation, UpdateEntryPoint},
        searcher::ConnectPlanV,
        GraphMem, VectorStore,
    },
};
use eyre::{eyre, Result};
use futures::future::try_join_all;
use futures::StreamExt;
use iris_mpc_common::{postgres::PostgresClient, vector_id::VectorId};
use serde::{de::DeserializeOwned, Serialize};
use sqlx::{error::BoxDynError, types::Json, PgConnection, Postgres, Row, Transaction};
use std::{collections::BTreeMap, marker::PhantomData, ops::DerefMut, str::FromStr};
use tokio::sync::mpsc;

#[derive(sqlx::FromRow, Debug, Clone, PartialEq, Eq)]
pub struct GenesisGraphCheckpointRow {
    pub id: i64,
    pub s3_key: String,
    pub last_indexed_iris_id: i64,
    pub last_indexed_modification_id: i64,
    pub blake3_hash: String,
    pub graph_version: i32,
    pub is_archival: bool,
}

/// A row from the hawk_graph_mutations table.
/// The serialized_mutation field contains a bincode-serialized BothEyes<GraphMutation<VectorId>>,
/// storing graph mutations for both left and right eyes in a single row.
#[derive(sqlx::FromRow, Debug, Clone, PartialEq, Eq)]
pub struct GraphMutationRow {
    pub id: i64,
    pub modification_id: i64,
    /// Bincode-serialized BothEyes<Vec<GraphMutation<VectorId>>> (mutations for both eyes)
    pub serialized_mutation: Vec<u8>,
}

pub struct GraphPg<V: VectorStore> {
    pool: sqlx::PgPool,
    schema_name: String,
    phantom: PhantomData<V>,
}

impl<V: VectorStore> GraphPg<V> {
    pub async fn new(postgres_client: &PostgresClient) -> Result<Self> {
        tracing::info!(
            "Created a graph store with schema: {}",
            postgres_client.schema_name,
        );

        postgres_client.migrate().await;

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

    pub async fn insert_genesis_graph_checkpoint(
        tx: &mut Transaction<'_, Postgres>,
        s3_key: &str,
        last_indexed_iris_id: i64,
        last_indexed_modification_id: i64,
        blake3_hash: &str,
        is_archival: bool,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO genesis_graph_checkpoint (
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                blake3_hash,
                is_archival
            )
            VALUES ($1, $2, $3, $4, $5)
            "#,
        )
        .bind(s3_key)
        .bind(last_indexed_iris_id)
        .bind(last_indexed_modification_id)
        .bind(blake3_hash)
        .bind(is_archival)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Returns the most recent genesis graph checkpoint
    pub async fn get_latest_genesis_graph_checkpoint(
        &self,
    ) -> Result<Option<GenesisGraphCheckpointRow>> {
        let row = sqlx::query_as::<_, GenesisGraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_version,
                blake3_hash,
                is_archival
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
    ) -> Result<Option<GenesisGraphCheckpointRow>> {
        let row = sqlx::query_as::<_, GenesisGraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_version,
                blake3_hash,
                is_archival
            FROM genesis_graph_checkpoint
            WHERE s3_key = $1
            "#,
        )
        .bind(s3_key)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Returns genesis graph checkpoints in descending order
    pub async fn get_genesis_graph_checkpoints(&self) -> Result<Vec<GenesisGraphCheckpointRow>> {
        let rows = sqlx::query_as::<_, GenesisGraphCheckpointRow>(
            r#"
            SELECT
                id,
                s3_key,
                last_indexed_iris_id,
                last_indexed_modification_id,
                graph_version,
                blake3_hash,
                is_archival
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
        let _ = sqlx::query_as::<_, GenesisGraphCheckpointRow>(
            r#"
            DELETE FROM genesis_graph_checkpoint WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| eyre!("Failed to delete genesis checkpoint: {e}"))?;

        Ok(())
    }

    /// Copies the graph-related tables to backup tables with a `_backup` suffix in the same schema.
    ///
    /// Tables backed up:
    /// - `hawk_graph_mutations`
    /// - `persistent_state`
    /// - `genesis_graph_checkpoint`
    ///
    /// This is a destructive operation for the backup tables: any existing data in them will be lost.
    ///
    /// Example usage:
    ///     graph_pg.backup_hawk_graph_tables().await?;
    pub async fn backup_hawk_graph_tables(&self) -> Result<()> {
        let schema = &self.schema_name;
        let tables = [
            "hawk_graph_mutations",
            "persistent_state",
            "genesis_graph_checkpoint",
        ];

        for table_name in tables {
            let src_table = format!("\"{}\".{}", schema, table_name);
            let backup_table = format!("\"{}\".{}_backup", schema, table_name);

            // Drop backup table if exists
            sqlx::query(&format!("DROP TABLE IF EXISTS {backup_table} CASCADE"))
                .execute(&self.pool)
                .await?;
            // Create backup table as a copy of the source
            sqlx::query(&format!("CREATE TABLE {backup_table} AS TABLE {src_table}"))
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    /// Restores graph-related tables from their backup tables in the same schema.
    ///
    /// Tables restored:
    /// - `hawk_graph_mutations`
    /// - `persistent_state`
    /// - `genesis_graph_checkpoint`
    ///
    /// This is a destructive operation for the main tables: any existing data in them will be lost.
    ///
    /// Example usage:
    ///     graph_pg.restore_hawk_graph_tables_from_backup().await?;
    pub async fn restore_hawk_graph_tables_from_backup(&self) -> Result<()> {
        let schema = &self.schema_name;
        let tables = [
            "hawk_graph_mutations",
            "persistent_state",
            "genesis_graph_checkpoint",
        ];

        for table_name in tables {
            let main_table = format!("\"{}\".{}", schema, table_name);
            let backup_table = format!("\"{}\".{}_backup", schema, table_name);

            // Drop main table if it exists
            sqlx::query(&format!("DROP TABLE IF EXISTS {main_table} CASCADE"))
                .execute(&self.pool)
                .await?;

            // Recreate main table from backup
            sqlx::query(&format!(
                "CREATE TABLE {main_table} AS TABLE {backup_table}"
            ))
            .execute(&self.pool)
            .await?;
        }

        Ok(())
    }

    pub async fn insert_hawk_graph_mutations(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        modification_id_height: i64,
        serialized_mutations: &[Vec<u8>],
    ) -> Result<Vec<GraphMutationRow>> {
        if serialized_mutations.is_empty() {
            return Ok(Vec::new());
        }

        let rows = sqlx::query_as::<_, GraphMutationRow>(
            r#"
            INSERT INTO hawk_graph_mutations (modification_id, serialized_mutation)
            SELECT $1, unnest($2::bytea[])
            RETURNING id, modification_id, serialized_mutation
            "#,
        )
        .bind(modification_id_height)
        .bind(&serialized_mutations)
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
            SELECT id, modification_id, serialized_mutation
            FROM hawk_graph_mutations
            WHERE $1::bigint IS NULL OR id <= $1
            ORDER BY id ASC
            "#,
        )
        .bind(max_graph_mutation_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().collect())
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
                    WHERE id <= $1
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
}

pub struct GraphTx<'a, V> {
    pub tx: Transaction<'a, Postgres>,
    schema_name: String,
    phantom: PhantomData<V>,
}

impl<'b, V: VectorStore> GraphTx<'b, V> {
    pub fn with_graph<'a>(&'a mut self, graph_id: StoreId) -> GraphOps<'a, 'b, V> {
        GraphOps { tx: self, graph_id }
    }

    /// Insert a single graph mutation row into hawk_graph_mutations.
    /// Returns the generated mutation id.
    pub async fn insert_hawk_graph_mutation(
        &mut self,
        modification_id: i64,
        serialized_mutation: &[u8],
    ) -> Result<i64> {
        let row = sqlx::query(
            r#"
            INSERT INTO hawk_graph_mutations (modification_id, serialized_mutation)
            VALUES ($1, $2)
            RETURNING id
            "#,
        )
        .bind(modification_id)
        .bind(serialized_mutation)
        .fetch_one(self.tx.deref_mut())
        .await?;

        Ok(row.get("id"))
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
}

pub struct GraphOps<'a, 'b, V> {
    tx: &'a mut GraphTx<'b, V>,
    graph_id: StoreId,
}

// todo sw: perhaps delete these too
impl<V: VectorStore<VectorRef = VectorId>> GraphOps<'_, '_, V> {
    fn graph_id(&self) -> i16 {
        self.graph_id as i16
    }

    fn tx(&mut self) -> &mut PgConnection {
        self.tx.tx.deref_mut()
    }
}

pub mod test_utils {
    use super::*;
    use iris_mpc_common::postgres::{AccessMode, PostgresClient};
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
    use std::sync::Arc;

    use super::{test_utils::TestGraphPg, *};
    use crate::{
        hawkers::{aby3::aby3_store::FhdOps, plaintext_store::PlaintextStore},
        hnsw::{
            graph::{layered_graph::EntryPoint, neighborhood::Neighborhood},
            vector_store::VectorStoreMut,
            GraphMem, HnswSearcher, SortedNeighborhood,
        },
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
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

    #[tokio::test]
    async fn test_backup_hawk_graph_tables_creates_and_copies() -> Result<()> {
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let pool = graph.pool();
        let schema = &graph.graph.schema_name;
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        let entry_backup = format!("\"{}\".hawk_graph_entry_backup", schema);

        // Insert a row into hawk_graph_entry
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer) VALUES ($1, $2, $3, $4)",
            entry_table
        ))
        .bind(1i16)
        .bind(42i64)
        .bind(0i16)
        .bind(0i16)
        .execute(pool)
        .await?;

        // Run backup
        graph.graph.backup_hawk_graph_tables().await?;

        // Check backup table exists and has the data
        let entry_row: Option<(i16, i64, i16, i16)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer FROM {}",
            entry_backup
        ))
        .fetch_optional(pool)
        .await?;
        assert!(entry_row.is_some(), "Entry backup row should exist");
        let (graph_id, serial_id, version_id, layer) = entry_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 42);
        assert_eq!(version_id, 0);
        assert_eq!(layer, 0);

        // Clean up
        graph.cleanup().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_backup_hawk_graph_tables_overwrites_existing_backup() -> Result<()> {
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let pool = graph.pool();
        let schema = &graph.graph.schema_name;
        let entry_backup = format!("\"{}\".hawk_graph_entry_backup", schema);

        // Create backup table with dummy data
        sqlx::query(&format!("CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2)", entry_backup)).execute(pool).await?;
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer) VALUES ($1, $2, $3, $4)",
            entry_backup
        ))
        .bind(0i16)
        .bind(99i64)
        .bind(99i16)
        .bind(99i16)
        .execute(pool)
        .await?;

        // Insert a row into hawk_graph_entry
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer) VALUES ($1, $2, $3, $4)",
            entry_table
        ))
        .bind(1i16)
        .bind(43i64)
        .bind(1i16)
        .bind(1i16)
        .execute(pool)
        .await?;

        // Run backup (should overwrite backup table)
        graph.graph.backup_hawk_graph_tables().await?;

        // Check backup table has only the new data
        let entry_row: Option<(i16, i64, i16, i16)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer FROM {}",
            entry_backup
        ))
        .fetch_optional(pool)
        .await?;
        assert!(entry_row.is_some(), "Entry backup row should exist");
        let (graph_id, serial_id, version_id, layer) = entry_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 43);
        assert_eq!(version_id, 1);
        assert_eq!(layer, 1);

        // Clean up
        graph.cleanup().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_restore_hawk_graph_tables_from_backup() -> Result<()> {
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let pool = graph.pool();
        let schema = &graph.graph.schema_name;
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        let entry_backup = format!("\"{}\".hawk_graph_entry_backup", schema);

        // Insert a row into backup table
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2)", entry_backup
        )).execute(pool).await?;
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer) VALUES ($1, $2, $3, $4)",
            entry_backup
        ))
        .bind(1i16)
        .bind(99i64)
        .bind(2i16)
        .bind(3i16)
        .execute(pool)
        .await?;

        // Overwrite main table with dummy data
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer) VALUES ($1, $2, $3, $4)",
            entry_table
        ))
        .bind(0i16)
        .bind(1i64)
        .bind(1i16)
        .bind(1i16)
        .execute(pool)
        .await?;

        // Now restore from backup
        graph.graph.restore_hawk_graph_tables_from_backup().await?;

        // Check that main table now has the backup data
        let entry_row: Option<(i16, i64, i16, i16)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer FROM {}",
            entry_table
        ))
        .fetch_optional(pool)
        .await?;
        assert!(entry_row.is_some(), "Entry row should exist after restore");
        let (graph_id, serial_id, version_id, layer) = entry_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 99);
        assert_eq!(version_id, 2);
        assert_eq!(layer, 3);

        graph.cleanup().await?;
        Ok(())
    }
}
