use super::neighborhood::SortedEdgeIds;
use crate::{
    execution::hawk_main::StoreId,
    hnsw::{
        searcher::{ConnectPlanLayerV, ConnectPlanV},
        GraphMem, VectorStore,
    },
};
use eyre::{eyre, Result};
use futures::{Stream, StreamExt, TryStreamExt};
use iris_mpc_common::{postgres::PostgresClient, vector_id::VectorId};
use itertools::izip;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sqlx::{
    error::BoxDynError, types::Json, types::Text, PgConnection, Postgres, Row, Transaction,
};
use std::{marker::PhantomData, ops::DerefMut, str::FromStr};

#[derive(sqlx::FromRow, Debug, PartialEq, Eq)]
pub struct RowLinks {
    serial_id: i64,
    version_id: i16,
    // this is a serialized SortedEdgeIds<VectorId> (using bincode)
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
}

pub struct GraphTx<'a, V> {
    pub tx: Transaction<'a, Postgres>,
    schema_name: String,
    phantom: PhantomData<V>,
}

impl<'b, V: VectorStore> GraphTx<'b, V> {
    pub fn with_graph<'a>(&'a mut self, graph_id: StoreId) -> GraphOps<'a, 'b, V> {
        GraphOps {
            tx: self,
            graph_id,
            borrowable_sql: "".to_string(),
        }
    }
}

pub struct GraphOps<'a, 'b, V> {
    tx: &'a mut GraphTx<'b, V>,
    graph_id: StoreId,
    borrowable_sql: String,
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
        if plan.set_ep {
            let insertion_layer = plan.layers.len() - 1;
            self.set_entry_point(plan.inserted_vector, insertion_layer)
                .await?;
        }

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_plan) in plan.layers.into_iter().enumerate() {
            self.connect_apply(plan.inserted_vector, lc, layer_plan)
                .await?;
        }

        Ok(())
    }

    /// Apply the connections from `HnswSearcher::connect_prepare` to the graph.
    async fn connect_apply(
        &mut self,
        q: V::VectorRef,
        lc: usize,
        plan: ConnectPlanLayerV<V>,
    ) -> Result<()> {
        // Connect all n -> q.
        for ((n, _nq), links) in izip!(plan.neighbors.iter(), plan.nb_links) {
            self.set_links(*n, links, lc).await?;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors.edge_ids(), lc).await?;

        Ok(())
    }

    pub async fn get_entry_point(&mut self) -> Result<Option<(V::VectorRef, usize)>> {
        let table = self.entry_table();
        let opt = sqlx::query(&format!(
            "SELECT serial_id, version_id, layer FROM {table} WHERE graph_id = $1"
        ))
        .bind(self.graph_id())
        .fetch_optional(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch entry point: {e}"))?;

        if let Some(row) = opt {
            let serial_id: i64 = row.get("serial_id");
            let version_id: i16 = row.get("version_id");
            let layer: i16 = row.get("layer");

            Ok(Some((
                VectorId::new(serial_id as u32, version_id),
                layer as usize,
            )))
        } else {
            Ok(None)
        }
    }

    pub async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) -> Result<()> {
        let table = self.entry_table();
        sqlx::query(&format!(
            "
            INSERT INTO {table} (graph_id, serial_id, version_id, layer)
            VALUES ($1, $2, $3, $4) ON CONFLICT (graph_id)
            DO UPDATE SET (serial_id, version_id, layer) = (EXCLUDED.serial_id, EXCLUDED.version_id, EXCLUDED.layer)
            "
        ))
        .bind(self.graph_id())
        .bind(point.serial_id() as i64)
        .bind(point.version_id())
        .bind(layer as i16)
        .execute(self.tx())
        .await
        .map_err(|e| eyre!("Failed to set entry point: {e}"))?;

        Ok(())
    }

    pub async fn get_links(
        &mut self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> Result<SortedEdgeIds<V::VectorRef>> {
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
            Ok(SortedEdgeIds::default())
        }
    }

    pub async fn set_links(
        &mut self,
        base: V::VectorRef,
        links: SortedEdgeIds<V::VectorRef>,
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
}

impl<V: VectorStore<VectorRef = VectorId>> GraphOps<'_, '_, V>
where
    BoxDynError: From<<V::VectorRef as FromStr>::Err>,
    V::DistanceRef: Send + Unpin + 'static,
{
    fn stream_links(&mut self) -> impl Stream<Item = Result<RowLinks>> + '_ {
        let table = self.links_table();
        self.borrowable_sql = format!(
            "
            SELECT serial_id, version_id, links, layer FROM {table}
            WHERE graph_id = $1
            "
        );
        sqlx::query_as::<_, RowLinks>(&self.borrowable_sql)
            .bind(self.graph_id())
            .fetch(self.tx.tx.deref_mut())
            .map_err(Into::into)
    }

    pub async fn load_to_mem(&mut self) -> Result<GraphMem<V>> {
        let mut graph_mem = GraphMem::new();

        let ep = self.get_entry_point().await?;

        if let Some((point, layer)) = ep {
            graph_mem.set_entry_point(point, layer).await;
        }

        let mut count = 0;

        let mut irises = self.stream_links();
        while let Some(row) = irises.next().await {
            let row = row?;
            let links = bincode::deserialize(&row.links)?;
            let source_ref = VectorId::new(row.serial_id as u32, row.version_id);
            graph_mem
                .set_links(source_ref, links, row.layer as usize)
                .await;
            count += 1;
            if count % 100000 == 0 {
                tracing::info!("GraphLoader: Loaded {} graph links", count);
            }
        }

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
            graph::layered_graph::EntryPoint, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
            SortedNeighborhood,
        },
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tokio;

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
            let q = vector_store.points[0].clone();
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
                    .insert(&mut vector_store, vectors[j], distances[j])
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
        let db = HnswSearcher::new_with_test_parameters();

        let queries1 = IrisDB::new_random_rng(10, rng)
            .db
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();

        // Insert the codes.
        let mut tx = graph_pg.tx().await.unwrap();
        for query in queries1.iter() {
            let insertion_layer = db.select_layer(rng)?;
            let (neighbors, set_ep) = db
                .search_to_insert(vector_store, graph_mem, query, insertion_layer)
                .await?;
            assert!(!db.is_match(vector_store, &neighbors).await?);
            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            let plan = db
                .insert_prepare(vector_store, graph_mem, inserted, neighbors, set_ep)
                .await?;

            graph_mem.insert_apply(plan.clone()).await;
            tx.with_graph(StoreId::Left).insert_apply(plan).await?;
        }

        let graph_mem2 = tx.with_graph(StoreId::Left).load_to_mem().await?;
        assert_eq!(graph_mem, &graph_mem2);

        // Search for the same codes and find matches.
        for query in queries1.iter() {
            let neighbors = db.search(vector_store, graph_mem, query, 1).await?;
            assert!(db.is_match(vector_store, &[neighbors]).await?);
        }

        tx.tx.commit().await.unwrap();
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
}
