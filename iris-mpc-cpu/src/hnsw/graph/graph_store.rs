use super::{layered_graph::EntryPoint, neighborhood::SortedEdgeIds};
use crate::{
    execution::hawk_main::StoreId,
    hnsw::{
        searcher::{ConnectPlanLayerV, ConnectPlanV},
        GraphMem, VectorStore,
    },
};
use eyre::{eyre, Result};
use futures::{Stream, StreamExt, TryStreamExt};
use iris_mpc_common::postgres::PostgresClient;
use itertools::izip;
use serde::{Deserialize, Serialize};
use sqlx::{error::BoxDynError, types::Text, PgConnection, Postgres, Row, Transaction};
use std::{marker::PhantomData, ops::DerefMut, str::FromStr};

#[derive(sqlx::FromRow, Debug, PartialEq, Eq)]
pub struct RowLinks<V: VectorStore> {
    source_ref: Text<V::VectorRef>,
    // this is a serialized SortedEdgeIds<V::VectorRef> (using bincode)
    links: Vec<u8>,
    layer: i32,
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

impl<V: VectorStore> GraphOps<'_, '_, V>
where
    V::VectorRef: Sized + Serialize + for<'a> Deserialize<'a>,
{
    fn entry_table(&self) -> String {
        format!("\"{}\".hawk_graph_entry", self.tx.schema_name)
    }

    fn links_table(&self) -> String {
        format!("\"{}\".hawk_graph_links", self.tx.schema_name)
    }

    fn graph_id(&self) -> i32 {
        self.graph_id as i32
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
            self.set_entry_point(plan.inserted_vector.clone(), insertion_layer)
                .await?;
        }

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_plan) in plan.layers.into_iter().enumerate() {
            self.connect_apply(plan.inserted_vector.clone(), lc, layer_plan)
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
            self.set_links(n.clone(), links, lc).await?;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors.edge_ids(), lc).await?;

        Ok(())
    }

    pub async fn get_entry_point(&mut self) -> Result<Option<(V::VectorRef, usize)>> {
        let table = self.entry_table();
        let opt = sqlx::query(&format!(
            "SELECT entry_point FROM {table} WHERE graph_id = $1"
        ))
        .bind(self.graph_id())
        .fetch_optional(self.tx())
        .await
        .map_err(|e| eyre!("Failed to fetch entry point: {e}"))?;

        if let Some(row) = opt {
            let entry_point: Vec<u8> = row.get("entry_point");
            let x: EntryPoint<V::VectorRef> = bincode::deserialize(&entry_point)
                .map_err(|e| eyre!("Failed to deserialize entry point: {e}"))?;
            Ok(Some((x.point, x.layer)))
        } else {
            Ok(None)
        }
    }

    pub async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) -> Result<()> {
        let entry_buf = EntryPoint { point, layer }
            .to_packed()
            .map_err(|e| eyre!("Failed to serialize entry point: {e}"))?;

        let table = self.entry_table();
        sqlx::query(&format!(
            "
            INSERT INTO {table} (graph_id, entry_point)
            VALUES ($1, $2) ON CONFLICT (graph_id)
            DO UPDATE SET entry_point = EXCLUDED.entry_point"
        ))
        .bind(self.graph_id())
        .bind(entry_buf)
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
            WHERE graph_id = $1 AND source_ref = $2 AND layer = $3
        "
        ))
        .bind(self.graph_id())
        .bind(Text(base))
        .bind(lc as i32)
        .fetch_optional(self.tx())
        .await?;

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
        let links = links
            .to_packed()
            .map_err(|e| eyre!("Failed to serialize links: {e}"))?;

        let table = self.links_table();
        sqlx::query(&format!(
            "
            INSERT INTO {table} (graph_id, source_ref, layer, links)
            VALUES ($1, $2, $3, $4) ON CONFLICT (graph_id, source_ref, layer)
            DO UPDATE SET
            links = EXCLUDED.links
        "
        ))
        .bind(self.graph_id())
        .bind(Text(base))
        .bind(lc as i32)
        .bind(links)
        .execute(self.tx())
        .await
        .map_err(|e| eyre!("Failed to set links: {e}"))?;

        Ok(())
    }
}

impl<V: VectorStore> GraphOps<'_, '_, V>
where
    V::VectorRef: Serialize + for<'a> Deserialize<'a> + Send + Unpin + 'static,
    BoxDynError: From<<V::VectorRef as FromStr>::Err>,
    V::DistanceRef: Send + Unpin + 'static,
{
    fn stream_links(&mut self) -> impl Stream<Item = Result<RowLinks<V>>> + '_ {
        let table = self.links_table();
        self.borrowable_sql = format!(
            "
            SELECT source_ref, links, layer FROM {table}
            WHERE graph_id = $1
        "
        );
        sqlx::query_as::<_, RowLinks<V>>(&self.borrowable_sql)
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
            graph_mem
                .set_links(row.source_ref.0, links, row.layer as usize)
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
        hnsw::{vector_store::VectorStoreMut, GraphMem, HnswSearcher, SortedNeighborhood},
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
}
