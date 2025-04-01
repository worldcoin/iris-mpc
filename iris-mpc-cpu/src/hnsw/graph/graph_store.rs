use super::layered_graph::EntryPoint;
use crate::{
    execution::hawk_main::StoreId,
    hnsw::{
        graph::neighborhood::SortedNeighborhoodV,
        searcher::{ConnectPlanLayerV, ConnectPlanV},
        GraphMem, VectorStore,
    },
};
use eyre::Result;
use futures::{Stream, StreamExt, TryStreamExt};
use iris_mpc_common::postgres::PostgresClient;
use itertools::izip;
use sqlx::{
    error::BoxDynError,
    postgres::PgRow,
    types::{Json, Text},
    PgConnection, Postgres, Row, Transaction,
};
use std::{marker::PhantomData, ops::DerefMut, str::FromStr};

#[derive(sqlx::FromRow, Debug, PartialEq, Eq)]
pub struct RowLinks<V: VectorStore> {
    source_ref: Text<V::VectorRef>,
    links: Json<SortedNeighborhoodV<V>>,
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

impl<V: VectorStore> GraphOps<'_, '_, V> {
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
    pub async fn insert_apply(&mut self, plan: ConnectPlanV<V>) {
        // If required, set vector as new entry point
        if plan.set_ep {
            let insertion_layer = plan.layers.len() - 1;
            self.set_entry_point(plan.inserted_vector.clone(), insertion_layer)
                .await;
        }

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_plan) in plan.layers.into_iter().enumerate() {
            self.connect_apply(plan.inserted_vector.clone(), lc, layer_plan)
                .await;
        }
    }

    /// Apply the connections from `HnswSearcher::connect_prepare` to the graph.
    async fn connect_apply(&mut self, q: V::VectorRef, lc: usize, plan: ConnectPlanLayerV<V>) {
        // Connect all n -> q.
        for ((n, _nq), links) in izip!(plan.neighbors.iter(), plan.nb_links) {
            self.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors, lc).await;
    }

    pub async fn get_entry_point(&mut self) -> Option<(V::VectorRef, usize)> {
        let table = self.entry_table();
        sqlx::query(&format!(
            "SELECT entry_point FROM {table} WHERE graph_id = $1"
        ))
        .bind(self.graph_id())
        .fetch_optional(self.tx())
        .await
        .expect("Failed to fetch entry point")
        .map(|row: PgRow| {
            let x: Json<EntryPoint<V::VectorRef>> = row.get("entry_point");
            (x.point.clone(), x.layer)
        })
    }

    async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) {
        let table = self.entry_table();
        sqlx::query(&format!(
            "
            INSERT INTO {table} (graph_id, entry_point)
            VALUES ($1, $2) ON CONFLICT (graph_id)
            DO UPDATE SET entry_point = EXCLUDED.entry_point"
        ))
        .bind(self.graph_id())
        .bind(Json(&EntryPoint { point, layer }))
        .execute(self.tx())
        .await
        .expect("Failed to set entry point");
    }

    pub async fn get_links(
        &mut self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> SortedNeighborhoodV<V> {
        let table = self.links_table();
        sqlx::query(&format!(
            "
            SELECT links FROM {table}
            WHERE graph_id = $1 AND source_ref = $2 AND layer = $3
        "
        ))
        .bind(self.graph_id())
        .bind(Text(base))
        .bind(lc as i32)
        .fetch_optional(self.tx())
        .await
        .expect("Failed to fetch links")
        .map(|row: PgRow| {
            let x: Json<SortedNeighborhoodV<V>> = row.get("links");
            x.as_ref().clone()
        })
        .unwrap_or_else(SortedNeighborhoodV::<V>::new)
    }

    async fn set_links(&mut self, base: V::VectorRef, links: SortedNeighborhoodV<V>, lc: usize) {
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
        .bind(Json(&links))
        .execute(self.tx())
        .await
        .expect("Failed to set links");
    }
}

impl<V: VectorStore> GraphOps<'_, '_, V>
where
    V::VectorRef: Send + Unpin + 'static,
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

        let ep = self.get_entry_point().await;

        if let Some((point, layer)) = ep {
            graph_mem.set_entry_point(point, layer).await;
        }

        let mut irises = self.stream_links();
        while let Some(row) = irises.next().await {
            let row = row?;
            graph_mem
                .set_links(row.source_ref.0, row.links.0, row.layer as usize)
                .await;
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
            let postgres_client = PostgresClient::new(&test_db_url()?, &schema_name, AccessMode::ReadWrite).await?;
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
    use crate::{
        hawkers::plaintext_store::PlaintextStore,
        hnsw::{vector_store::VectorStoreMut, GraphMem, HnswSearcher, SortedNeighborhood},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_db() {
        let graph = TestGraphPg::<PlaintextStore>::new().await.unwrap();
        let mut vector_store = PlaintextStore::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);

        let vectors = {
            let mut v = vec![];
            for raw_query in IrisDB::new_random_rng(10, rng).db {
                let q = vector_store.prepare_query(raw_query);
                v.push(vector_store.insert(&q).await);
            }
            v
        };

        let distances = {
            let mut d = vec![];
            for v in vectors.iter() {
                d.push(vector_store.eval_distance(&vectors[0], v).await);
            }
            d
        };

        let mut tx = graph.tx().await.unwrap();
        let mut graph_ops = tx.with_graph(StoreId::Left);

        let ep = graph_ops.get_entry_point().await;
        assert!(ep.is_none());

        let ep2 = EntryPoint {
            point: vectors[0],
            layer: ep.map(|e| e.1).unwrap_or_default() + 1,
        };

        graph_ops.set_entry_point(ep2.point, ep2.layer).await;

        let (point3, layer3) = graph_ops.get_entry_point().await.unwrap();
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
                    .await;
            }

            graph_ops.set_links(vectors[i], links.clone(), 0).await;

            let links2 = graph_ops.get_links(&vectors[i], 0).await;
            assert_eq!(*links, *links2);
        }

        tx.tx.commit().await.unwrap();
        graph.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_hnsw_db() {
        let graph_pg = TestGraphPg::<PlaintextStore>::new().await.unwrap();
        let graph_mem = &mut GraphMem::new();
        let vector_store = &mut PlaintextStore::default();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let db = HnswSearcher::default();

        let queries1 = IrisDB::new_random_rng(10, rng)
            .db
            .into_iter()
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        let mut tx = graph_pg.tx().await.unwrap();
        for query in queries1.iter() {
            let insertion_layer = db.select_layer(rng);
            let (neighbors, set_ep) = db
                .search_to_insert(vector_store, graph_mem, query, insertion_layer)
                .await;
            assert!(!db.is_match(vector_store, &neighbors).await);
            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            let plan = db
                .insert_prepare(vector_store, graph_mem, inserted, neighbors, set_ep)
                .await;

            graph_mem.insert_apply(plan.clone()).await;
            tx.with_graph(StoreId::Left).insert_apply(plan).await;
        }

        let graph_mem2 = tx.with_graph(StoreId::Left).load_to_mem().await.unwrap();
        assert_eq!(graph_mem, &graph_mem2);

        // Search for the same codes and find matches.
        for query in queries1.iter() {
            let neighbors = db.search(vector_store, graph_mem, query, 1).await;
            assert!(db.is_match(vector_store, &[neighbors]).await);
        }

        tx.tx.commit().await.unwrap();
        graph_pg.cleanup().await.unwrap();
    }
}
