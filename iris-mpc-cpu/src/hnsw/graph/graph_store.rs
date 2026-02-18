use crate::{
    execution::hawk_main::StoreId,
    hnsw::{
        searcher::{ConnectPlanV, UpdateEntryPoint},
        GraphMem, VectorStore,
    },
};
use eyre::{eyre, Result};
use futures::future::try_join_all;
use futures::StreamExt;
use iris_mpc_common::{postgres::PostgresClient, vector_id::VectorId};
use serde::{de::DeserializeOwned, Serialize};
use sqlx::{error::BoxDynError, types::Json, PgConnection, Postgres, Row, Transaction};
use std::{collections::HashMap, marker::PhantomData, ops::DerefMut, str::FromStr};
use tokio::sync::mpsc;

#[derive(sqlx::FromRow, Debug, PartialEq, Eq)]
pub struct RowLinks {
    serial_id: i64,
    version_id: i16,
    // this is a serialized Vec<VectorId> (using bincode)
    links: Vec<u8>,
    layer: i16,
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

    /// Copies the `hawk_graph_entry` and `hawk_graph_links` tables to backup tables with a `_backup` suffix in the same schema.
    ///
    /// - Drops the backup tables (`hawk_graph_entry_backup`, `hawk_graph_links_backup`) if they exist.
    /// - Recreates the backup tables with the same structure as the originals.
    /// - Copies all data from the original tables to the backup tables.
    ///
    /// This is a destructive operation for the backup tables: any existing data in them will be lost.
    ///
    /// Example usage:
    ///     graph_pg.backup_hawk_graph_tables().await?;
    pub async fn backup_hawk_graph_tables(&self) -> Result<()> {
        let schema = &self.schema_name;
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        let links_table = format!("\"{}\".hawk_graph_links", schema);
        let backup_entry = format!("\"{}\".hawk_graph_entry_backup", schema);
        let backup_links = format!("\"{}\".hawk_graph_links_backup", schema);

        // Drop backup tables if they exist, then create and populate in one step
        for (src, dest) in [
            (entry_table.as_str(), backup_entry.as_str()),
            (links_table.as_str(), backup_links.as_str()),
        ] {
            // Drop backup table if exists
            sqlx::query(&format!("DROP TABLE IF EXISTS {dest} CASCADE"))
                .execute(&self.pool)
                .await?;
            // Create backup table as a copy of the source (fastest for stable schema)
            sqlx::query(&format!("CREATE TABLE {dest} AS TABLE {src}"))
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    /// Restores the `hawk_graph_entry` and `hawk_graph_links` tables from their backup tables in the same schema.
    ///
    /// - Drops the main tables (`hawk_graph_entry`, `hawk_graph_links`) if they exist.
    /// - Recreates the main tables with the data from the backup tables (`hawk_graph_entry_backup`, `hawk_graph_links_backup`).
    /// - This is a destructive operation for the main tables: any existing data in them will be lost.
    ///
    /// Example usage:
    ///     graph_pg.restore_hawk_graph_tables_from_backup().await?;
    pub async fn restore_hawk_graph_tables_from_backup(&self) -> Result<()> {
        let schema = &self.schema_name;
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        let links_table = format!("\"{}\".hawk_graph_links", schema);
        let backup_entry = format!("\"{}\".hawk_graph_entry_backup", schema);
        let backup_links = format!("\"{}\".hawk_graph_links_backup", schema);

        // Drop main tables if they exist
        sqlx::query(&format!("DROP TABLE IF EXISTS {entry_table} CASCADE"))
            .execute(&self.pool)
            .await?;
        sqlx::query(&format!("DROP TABLE IF EXISTS {links_table} CASCADE"))
            .execute(&self.pool)
            .await?;

        // Recreate main tables from backup
        sqlx::query(&format!(
            "CREATE TABLE {entry_table} AS TABLE {backup_entry}"
        ))
        .execute(&self.pool)
        .await?;
        sqlx::query(&format!(
            "CREATE TABLE {links_table} AS TABLE {backup_links}"
        ))
        .execute(&self.pool)
        .await?;

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
}

pub struct GraphOps<'a, 'b, V> {
    tx: &'a mut GraphTx<'b, V>,
    graph_id: StoreId,
}

impl<V: VectorStore<VectorRef = VectorId>> GraphOps<'_, '_, V> {
    fn entry_table(&self) -> String {
        format!("\"{}\".hawk_graph_entry", self.tx.schema_name)
    }

    fn links_table(&self) -> String {
        format!("\"{}\".hawk_graph_links", self.tx.schema_name)
    }

    fn graph_id(&self) -> i16 {
        self.graph_id as i16
    }

    fn tx(&mut self) -> &mut PgConnection {
        self.tx.tx.deref_mut()
    }

    /// Apply an insertion plan from `HnswSearcher::insert_prepare` to the
    /// graph.
    pub async fn insert_apply(&mut self, plan: ConnectPlanV<V>) -> Result<()> {
        // If required, set vector as new entry point
        match plan.update_ep {
            UpdateEntryPoint::False => {}
            UpdateEntryPoint::SetUnique { layer } => {
                self.set_entry_point(plan.inserted_vector, layer).await?;
            }
            UpdateEntryPoint::Append { layer } => {
                self.add_entry_point(plan.inserted_vector, layer).await?;
            }
        }

        // Connect the new vector to its neighbors in each layer.
        for ((inserted_vector, lc), neighbors) in plan.updates {
            self.set_links(inserted_vector, neighbors, lc).await?;
        }

        Ok(())
    }

    pub async fn get_entry_point(&mut self) -> Result<Option<(V::VectorRef, usize)>> {
        Ok(self
            .get_entry_points()
            .await?
            .and_then(|(mut points, layer)| points.pop().map(|p| (p, layer))))
    }

    pub async fn get_entry_points(&mut self) -> Result<Option<(Vec<V::VectorRef>, usize)>> {
        let table = self.entry_table();
        let rows = sqlx::query(&format!(
            "SELECT serial_id, version_id, layer FROM {table} WHERE graph_id = $1"
        ))
        .bind(self.graph_id())
        .fetch_all(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch entry point: {e}"))?;

        if rows.is_empty() {
            Ok(None)
        } else {
            // ensure all entrypoints are on the same layer
            let mut expected_layer = None;
            let mut points = Vec::with_capacity(rows.len());
            for row in rows {
                let serial_id = row.get::<i64, &str>("serial_id") as u32;
                let version_id: i16 = row.get("version_id");
                let row_layer = row.get::<i16, &str>("layer") as usize;

                if expected_layer.is_none() {
                    expected_layer.replace(row_layer);
                }

                // if this fails, then add_entry_point() was used incorrectly.
                assert_eq!(Some(row_layer), expected_layer);

                points.push(VectorId::new(serial_id, version_id));
            }

            Ok(Some((points, expected_layer.unwrap())))
        }
    }

    pub async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) -> Result<()> {
        let table = self.entry_table();

        sqlx::query(&format!("DELETE FROM {table} WHERE graph_id = $1"))
            .bind(self.graph_id())
            .execute(self.tx())
            .await
            .map_err(|e| eyre!("Failed to clear entry points: {e}"))?;

        self.add_entry_point(point, layer).await?;

        Ok(())
    }

    pub async fn add_entry_point(&mut self, point: V::VectorRef, layer: usize) -> Result<()> {
        let table = self.entry_table();

        // insert the point into the table
        sqlx::query(&format!(
            "
                INSERT INTO {table} (graph_id, serial_id, version_id, layer)
                VALUES ($1, $2, $3, $4) ON CONFLICT (graph_id, serial_id, layer)
                DO UPDATE SET version_id = EXCLUDED.version_id
                "
        ))
        .bind(self.graph_id())
        .bind(point.serial_id() as i64)
        .bind(point.version_id())
        .bind(layer as i16)
        .execute(self.tx())
        .await
        .map_err(|e| eyre!("Failed to insert entry point: {e}"))?;

        Ok(())
    }

    pub async fn get_links(
        &mut self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> Result<Vec<V::VectorRef>> {
        let table = self.links_table();
        let opt = sqlx::query(&format!(
            "
            SELECT links FROM {table}
            WHERE graph_id = $1 AND serial_id = $2 AND version_id = $3 AND layer = $4
            "
        ))
        .bind(self.graph_id())
        .bind(base.serial_id() as i64)
        .bind(base.version_id())
        .bind(lc as i16)
        .fetch_optional(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch links: {e}"))?;

        if let Some(row) = opt {
            let links: Vec<u8> = row.get("links");
            bincode::deserialize(&links).map_err(|e| eyre!("Failed to deserialize links: {e}"))
        } else {
            Ok(Vec::default())
        }
    }

    pub async fn get_max_serial_id(&mut self) -> Result<i64> {
        let table = self.links_table();
        let row = sqlx::query(&format!(
            "SELECT MAX(serial_id) as max_serial_id FROM {table} WHERE graph_id = $1"
        ))
        .bind(self.graph_id())
        .fetch_one(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch largest serial id: {e}"))?;

        let max_serial_id: Option<i64> = row.try_get("max_serial_id").ok();
        Ok(max_serial_id.unwrap_or(0))
    }

    pub async fn set_links(
        &mut self,
        base: V::VectorRef,
        links: Vec<V::VectorRef>,
        lc: usize,
    ) -> Result<()> {
        let links =
            bincode::serialize(&links).map_err(|e| eyre!("Failed to serialize links: {e}"))?;

        let table = self.links_table();
        sqlx::query(&format!(
            "
            INSERT INTO {table} (graph_id, serial_id, version_id, layer, links)
            VALUES ($1, $2, $3, $4, $5) ON CONFLICT (graph_id, serial_id, version_id, layer)
            DO UPDATE SET
            links = EXCLUDED.links
            "
        ))
        .bind(self.graph_id())
        .bind(base.serial_id() as i64)
        .bind(base.version_id())
        .bind(lc as i16)
        .bind(links)
        .execute(self.tx())
        .await
        .map_err(|e| eyre!("Failed to set links: {e}"))?;

        Ok(())
    }

    pub async fn batch_set_links(
        &mut self,
        updates: HashMap<(i64, i16, i16), Vec<V::VectorRef>>,
    ) -> Result<()> {
        let mut serial_ids = Vec::with_capacity(updates.len());
        let mut version_ids = Vec::with_capacity(updates.len());
        let mut layers = Vec::with_capacity(updates.len());
        let mut links_blobs = Vec::with_capacity(updates.len());
        let graph_id = self.graph_id();

        #[allow(clippy::iter_over_hash_type, reason = "TODO")]
        for ((sid, vid, layer), neighbors) in updates {
            serial_ids.push(sid);
            version_ids.push(vid as i32);
            layers.push(layer);
            links_blobs.push(bincode::serialize(&neighbors)?);
        }

        let table = self.links_table();

        // We bind vectors of primitives.
        // sqlx maps Vec<T> to Postgres T[] automatically.
        sqlx::query(&format!(
            "INSERT INTO {table} (graph_id, serial_id, version_id, layer, links)
         SELECT $1, * FROM UNNEST($2::int8[], $3::int4[], $4::int2[], $5::bytea[])
         ON CONFLICT (graph_id, serial_id, version_id, layer)
         DO UPDATE SET links = EXCLUDED.links"
        ))
        .bind(graph_id) // $1: Single ID for the whole batch
        .bind(&serial_ids) // $2: Array of BIGINT
        .bind(&version_ids) // $3: Array of INTEGER
        .bind(&layers) // $4: Array of SMALLINT
        .bind(&links_blobs) // $5: Array of BYTEA
        .execute(self.tx())
        .await
        .map_err(|e| eyre!("Failed to batch set links: {e}"))?;

        Ok(())
    }

    /// Ensures that graph_entry and graph_links table are empty. For testing only
    pub async fn clear_tables(&mut self) -> Result<()> {
        let entry_table = self.entry_table();
        let links_table = self.links_table();

        sqlx::query(&format!("DELETE FROM {entry_table} WHERE graph_id = $1"))
            .bind(self.graph_id())
            .execute(self.tx())
            .await?;
        sqlx::query(&format!("DELETE FROM {links_table} WHERE graph_id = $1"))
            .bind(self.graph_id())
            .execute(self.tx())
            .await?;

        Ok(())
    }
}

impl<V: VectorStore<VectorRef = VectorId>> GraphOps<'_, '_, V>
where
    BoxDynError: From<<V::VectorRef as FromStr>::Err>,
    V::DistanceRef: Send + Unpin + 'static,
{
    /// Get total row count for this graph_id
    async fn get_total_row_count(&mut self) -> Result<usize> {
        let table = self.links_table();
        let row = sqlx::query(&format!(
            "
            SELECT COUNT(*) as total_count FROM {table}
            WHERE graph_id = $1
        "
        ))
        .bind(self.graph_id())
        .fetch_one(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch total row count: {e}"))?;

        Ok(row.get::<i64, _>("total_count") as usize)
    }

    /// Load graph data to memory using parallel loading by source_ref ranges.
    /// This method creates separate connections for each range and loads them in parallel.
    pub async fn load_to_mem(
        &mut self,
        pool: &sqlx::PgPool,
        parallelism: usize,
    ) -> Result<GraphMem<<V as VectorStore>::VectorRef>> {
        let mut graph_mem = GraphMem::new();
        let schema_name = self.tx.schema_name.clone();
        let graph_id = self.graph_id();

        let ep = self.get_entry_points().await?;
        if let Some((points, layer)) = ep {
            graph_mem.init_entry_points(points, layer).await;
        }

        let total_rows = self.get_total_row_count().await?;

        if total_rows == 0 {
            tracing::info!("GraphLoader: No data found for graph_id {}", graph_id);
            return Ok(graph_mem);
        }

        tracing::info!(
            "GraphLoader: Loading {} rows using {} parallel partitions for graph_id {}",
            total_rows,
            parallelism,
            graph_id
        );

        // Create channel for sending data from partitions to main task
        let (tx, mut rx) = mpsc::channel::<(V::VectorRef, Vec<V::VectorRef>, usize)>(1024);

        // Calculate partition size and create partition tasks
        let partition_size = total_rows.div_ceil(parallelism).max(1);

        let partition_tasks = (0..parallelism).map(|i| {
            let pool = pool.clone();
            let schema_name = schema_name.clone();
            let tx = tx.clone();

            async move {
                let mut conn = pool.acquire().await?;
                let table = format!("\"{}\".hawk_graph_links", schema_name);

                // Use OFFSET/LIMIT for partitioning by row count
                // We cannot rely on the serial id since there could be deletions
                let offset = partition_size * i;
                let limit = partition_size;

                let query_sql = format!(
                    "
                    SELECT serial_id, version_id, links, layer FROM {table}
                    WHERE graph_id = $1
                    ORDER BY serial_id, layer
                    OFFSET $2 LIMIT $3
                "
                );

                let mut rows = sqlx::query_as::<_, RowLinks>(&query_sql)
                    .bind(graph_id)
                    .bind(offset as i32)
                    .bind(limit as i32)
                    .fetch(&mut *conn);

                let mut links_loaded = 0;

                while let Some(row) = rows.next().await {
                    let row = row?;
                    let links = bincode::deserialize(&row.links)?;
                    let layer_idx = row.layer as usize;
                    let source_ref = VectorId::new(row.serial_id as u32, row.version_id);

                    // Send data to main task via channel
                    tx.send((source_ref, links, layer_idx))
                        .await
                        .map_err(|_| eyre!("Failed to send data to main task"))?;

                    links_loaded += 1;
                }

                tracing::info!(
                    "GraphLoader: Partition {} (IDs offset {} with limit {}) loaded {} links",
                    i,
                    offset,
                    limit,
                    links_loaded
                );

                Ok::<_, eyre::Error>(())
            }
        });

        // Start all partition tasks
        let partition_handles = partition_tasks.map(tokio::spawn).collect::<Vec<_>>();

        // Drop the original sender so rx will close when all partitions finish
        drop(tx);

        // Collect all data from partitions and write to graph_mem
        let mut total_links_written = 0;
        while let Some((source_ref, links, layer_idx)) = rx.recv().await {
            graph_mem.set_links(source_ref, links, layer_idx).await;
            total_links_written += 1;
            if total_links_written % 100000 == 0 {
                tracing::info!(
                    "GraphLoader: Loaded {} graph links on graph {}",
                    total_links_written,
                    self.graph_id()
                );
            }
        }

        // Wait for all partition tasks to complete and get their counts
        try_join_all(partition_handles)
            .await?
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        if total_links_written != total_rows {
            return Err(eyre!(
                "GraphLoader: Not all links were loaded. Expected {}, got {}",
                total_rows,
                total_links_written
            ));
        }

        tracing::info!(
            "GraphLoader: Loaded {} total graph links for graph_id {}",
            total_links_written,
            self.graph_id(),
        );

        Ok(graph_mem)
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
        hawkers::plaintext_store::PlaintextStore,
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
    async fn test_get_max_serial_id_empty() -> Result<()> {
        // Use the same pattern as other tests for setup/cleanup
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let mut tx = graph.tx().await?;
        let mut graph_ops = tx.with_graph(StoreId::Left);
        let max_id = graph_ops.get_max_serial_id().await?;
        assert_eq!(max_id, 0);
        tx.tx.commit().await?;
        graph.cleanup().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_get_largest_inserted_id_nonempty() -> Result<()> {
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let mut vector_store = PlaintextStore::new();
        let rng = &mut AesRng::seed_from_u64(42_u64);

        let vectors = {
            let mut v = vec![];
            for raw_query in IrisDB::new_random_rng(5, rng).db {
                let q = Arc::new(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        let distances = {
            let mut d = vec![];
            let q = vector_store
                .storage
                .get_vector_by_serial_id(1)
                .unwrap()
                .clone();
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&q, v).await?);
            }
            d
        };

        let mut tx = graph.tx().await?;
        let mut graph_ops = tx.with_graph(StoreId::Left);

        // Insert links for each vector, using the same pattern as test_db
        for i in 1..5 {
            let mut links = SortedNeighborhood::new();
            for j in 0..5 {
                if i != j {
                    links
                        .insert_and_trim(&mut vector_store, vectors[j], distances[j], links.len())
                        .await?;
                }
            }
            let links = links.edge_ids();
            graph_ops.set_links(vectors[i], links.clone(), 0).await?;
            let links2 = graph_ops.get_links(&vectors[i], 0).await?;
            assert_eq!(*links, *links2);
        }

        let max_id = graph_ops.get_max_serial_id().await?;
        // serial_id is stored as u32, but we inserted 5 vectors, so largest should be >= 4
        assert!(max_id >= 5);
        tx.tx.commit().await?;
        graph.cleanup().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_db() -> Result<()> {
        let graph = TestGraphPg::<PlaintextStore>::new().await?;
        let mut vector_store = PlaintextStore::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);

        let vectors = {
            let mut v = vec![];
            for raw_query in IrisDB::new_random_rng(10, rng).db {
                let q = Arc::new(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        let distances = {
            let mut d = vec![];
            let q = vector_store
                .storage
                .get_vector_by_serial_id(1)
                .unwrap()
                .clone();
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&q, v).await?);
            }
            d
        };

        let mut tx = graph.tx().await.unwrap();
        let mut graph_ops = tx.with_graph(StoreId::Left);

        let ep = graph_ops.get_entry_point().await?;
        assert!(ep.is_none());

        let ep2 = EntryPoint {
            point: vectors[0],
            layer: ep.map(|e| e.1).unwrap_or_default() + 1,
        };

        graph_ops.set_entry_point(ep2.point, ep2.layer).await?;

        let (point3, layer3) = graph_ops.get_entry_point().await?.unwrap();
        let ep3 = EntryPoint {
            point: point3,
            layer: layer3,
        };

        assert_eq!(ep2, ep3);

        for i in 1..4 {
            let mut links = SortedNeighborhood::new();

            for j in 4..7 {
                links
                    .insert_and_trim(&mut vector_store, vectors[j], distances[j], links.len() + 1)
                    .await?;
            }
            let links = links.edge_ids();

            graph_ops.set_links(vectors[i], links.clone(), 0).await?;

            let links2 = graph_ops.get_links(&vectors[i], 0).await?;
            assert_eq!(*links, *links2);
        }

        tx.tx.commit().await.unwrap();
        graph.cleanup().await.unwrap();

        Ok(())
    }

    #[tokio::test]
    async fn test_hnsw_db() -> Result<()> {
        let graph_pg = TestGraphPg::<PlaintextStore>::new().await?;
        let graph_mem = &mut GraphMem::new();
        let vector_store = &mut PlaintextStore::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let searcher = HnswSearcher::new_with_test_parameters();

        let queries1 = IrisDB::new_random_rng(10, rng)
            .db
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();
        // Insert the codes.
        let mut tx = graph_pg.tx().await?;
        for query in queries1.iter() {
            let insertion_layer = searcher.gen_layer_rng(rng)?;
            let (links, update_ep): (Vec<SortedNeighborhood<_>>, _) = searcher
                .search_to_insert(vector_store, graph_mem, query, insertion_layer)
                .await?;
            assert!(!searcher.is_match(vector_store, &links).await?);

            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;

            // Trim and extract unstructured vector lists
            let mut links_unstructured = Vec::new();
            for (lc, mut l) in links.iter().cloned().enumerate() {
                let m = searcher.params.get_M(lc);
                l.trim(vector_store, m).await?;
                links_unstructured.push(l.edge_ids())
            }

            let plan = searcher
                .insert_prepare(
                    vector_store,
                    graph_mem,
                    inserted,
                    links_unstructured,
                    update_ep,
                )
                .await?;

            graph_mem.insert_apply(plan.clone()).await;
            tx.with_graph(StoreId::Left).insert_apply(plan).await?;
        }
        tx.tx.commit().await?;

        // Test `get_total_row_count`
        let mut tx = graph_pg.tx().await?;
        let total_rows = tx.with_graph(StoreId::Left).get_total_row_count().await?;
        assert_eq!(total_rows, queries1.len() + 1);

        // Load graph using `load_to_mem`
        let graph_mem2 = tx
            .with_graph(StoreId::Left)
            .load_to_mem(graph_pg.pool(), 2)
            .await?;
        assert_eq!(graph_mem, &graph_mem2);

        // Search for the same codes and find matches.
        for query in queries1.iter() {
            let neighbors: SortedNeighborhood<_> =
                searcher.search(vector_store, &graph_mem2, query, 1).await?;
            assert!(searcher.is_match(vector_store, &[neighbors]).await?);
        }

        // Clean up
        tx.tx.commit().await?;
        graph_pg.cleanup().await.unwrap();

        Ok(())
    }

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
        let links_table = format!("\"{}\".hawk_graph_links", schema);
        let entry_backup = format!("\"{}\".hawk_graph_entry_backup", schema);
        let links_backup = format!("\"{}\".hawk_graph_links_backup", schema);

        // Insert a row into hawk_graph_entry and hawk_graph_links
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
        sqlx::query(&format!("INSERT INTO {} (graph_id, serial_id, version_id, layer, links) VALUES ($1, $2, $3, $4, $5)", links_table))
            .bind(1i16)
            .bind(42i64)
            .bind(0i16)
            .bind(0i16)
            .bind(vec![1u8,2u8,3u8])
            .execute(pool)
            .await?;

        // Run backup
        graph.graph.backup_hawk_graph_tables().await?;

        // Check backup tables exist and have the data
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

        let links_row: Option<(i16, i64, i16, i16, Vec<u8>)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer, links FROM {}",
            links_backup
        ))
        .fetch_optional(pool)
        .await?;
        assert!(links_row.is_some(), "Links backup row should exist");
        let (graph_id, serial_id, version_id, layer, links) = links_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 42);
        assert_eq!(version_id, 0);
        assert_eq!(layer, 0);
        assert_eq!(links, vec![1u8, 2u8, 3u8]);

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
        let links_backup = format!("\"{}\".hawk_graph_links_backup", schema);

        // Create backup tables with dummy data
        sqlx::query(&format!("CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2)", entry_backup)).execute(pool).await?;
        sqlx::query(&format!("CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2, links BYTEA)", links_backup)).execute(pool).await?;
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
        sqlx::query(&format!("INSERT INTO {} (graph_id, serial_id, version_id, layer, links) VALUES ($1, $2, $3, $4, $5)", links_backup))
            .bind(0i16)
            .bind(99i64)
            .bind(99i16)
            .bind(99i16)
            .bind(vec![9u8,9u8,9u8])
            .execute(pool)
            .await?;

        // Insert a row into hawk_graph_entry and hawk_graph_links
        let entry_table = format!("\"{}\".hawk_graph_entry", schema);
        let links_table = format!("\"{}\".hawk_graph_links", schema);
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
        sqlx::query(&format!("INSERT INTO {} (graph_id, serial_id, version_id, layer, links) VALUES ($1, $2, $3, $4, $5)", links_table))
            .bind(1i16)
            .bind(43i64)
            .bind(1i16)
            .bind(1i16)
            .bind(vec![4u8,5u8,6u8])
            .execute(pool)
            .await?;

        // Run backup (should overwrite backup tables)
        graph.graph.backup_hawk_graph_tables().await?;

        // Check backup tables have only the new data
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

        let links_row: Option<(i16, i64, i16, i16, Vec<u8>)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer, links FROM {}",
            links_backup
        ))
        .fetch_optional(pool)
        .await?;
        assert!(links_row.is_some(), "Links backup row should exist");
        let (graph_id, serial_id, version_id, layer, links) = links_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 43);
        assert_eq!(version_id, 1);
        assert_eq!(layer, 1);
        assert_eq!(links, vec![4u8, 5u8, 6u8]);

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
        let links_table = format!("\"{}\".hawk_graph_links", schema);
        let entry_backup = format!("\"{}\".hawk_graph_entry_backup", schema);
        let links_backup = format!("\"{}\".hawk_graph_links_backup", schema);

        // Insert a row into backup tables
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2)", entry_backup
        )).execute(pool).await?;
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2, links BYTEA)", links_backup
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
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer, links) VALUES ($1, $2, $3, $4, $5)", links_backup
        ))
        .bind(1i16)
        .bind(99i64)
        .bind(2i16)
        .bind(3i16)
        .bind(vec![7u8, 8u8, 9u8])
        .execute(pool)
        .await?;

        // Overwrite main tables with dummy data
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2)", entry_table
        )).execute(pool).await?;
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (graph_id INT2, serial_id INT8, version_id INT2, layer INT2, links BYTEA)", links_table
        )).execute(pool).await?;
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
        sqlx::query(&format!(
            "INSERT INTO {} (graph_id, serial_id, version_id, layer, links) VALUES ($1, $2, $3, $4, $5)", links_table
        ))
        .bind(0i16)
        .bind(1i64)
        .bind(1i16)
        .bind(1i16)
        .bind(vec![1u8, 2u8, 3u8])
        .execute(pool)
        .await?;

        // Now restore from backup
        graph.graph.restore_hawk_graph_tables_from_backup().await?;

        // Check that main tables now have the backup data
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

        let links_row: Option<(i16, i64, i16, i16, Vec<u8>)> = sqlx::query_as(&format!(
            "SELECT graph_id, serial_id, version_id, layer, links FROM {}",
            links_table
        ))
        .fetch_optional(pool)
        .await?;
        assert!(links_row.is_some(), "Links row should exist after restore");
        let (graph_id, serial_id, version_id, layer, links) = links_row.unwrap();
        assert_eq!(graph_id, 1);
        assert_eq!(serial_id, 99);
        assert_eq!(version_id, 2);
        assert_eq!(layer, 3);
        assert_eq!(links, vec![7u8, 8u8, 9u8]);

        graph.cleanup().await?;
        Ok(())
    }
}
