use super::layered_graph::EntryPoint;
use crate::hnsw::{
    graph::neighborhood::SortedNeighborhoodV,
    searcher::{ConnectPlanLayerV, ConnectPlanV},
    GraphMem, VectorStore,
};
use eyre::{eyre, Result};
use futures::{Stream, StreamExt, TryStreamExt};
use iris_mpc_common::config::Config;
use itertools::izip;
use sqlx::{
    error::BoxDynError,
    migrate::Migrator,
    postgres::{PgPoolOptions, PgRow},
    types::{Json, Text},
    Executor, Postgres, Row, Transaction,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    str::FromStr,
};

const APP_NAME: &str = "SMPC";
const MAX_CONNECTIONS: u32 = 5;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

#[derive(sqlx::FromRow, Debug, PartialEq, Eq)]
pub struct RowLinks<V: VectorStore> {
    source_ref: Text<V::VectorRef>,
    links:      Json<SortedNeighborhoodV<V>>,
    layer:      i32,
}

pub struct GraphPg<V: VectorStore> {
    pool:    sqlx::PgPool,
    phantom: PhantomData<V>,
}

pub struct GraphTx<'a, V> {
    pub tx:  Transaction<'a, Postgres>,
    phantom: PhantomData<V>,
}

impl<V: VectorStore> GraphPg<V> {
    /// Connect to a database based on Config URL, environment, and party_id.
    // TODO: Consolidate with Store?
    // TODO: Separate config?
    pub async fn new_from_config(config: &Config) -> Result<Self> {
        let db_config = config
            .database
            .as_ref()
            .ok_or(eyre!("Missing database config"))?;
        let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
        Self::new(&db_config.url, &schema_name).await
    }

    pub async fn new(url: &str, schema_name: &str) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema_name)?;

        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .after_connect(move |conn, _meta| {
                // Switch to the given schema in every connection.
                let connect_sql = connect_sql.clone();
                Box::pin(async move {
                    conn.execute(connect_sql.as_ref()).await.inspect_err(|e| {
                        eprintln!("error in after_connect: {:?}", e);
                    })?;
                    Ok(())
                })
            })
            .connect(url)
            .await?;

        // Create the schema on the first startup.
        MIGRATOR.run(&pool).await?;

        Ok(GraphPg {
            pool,
            phantom: PhantomData,
        })
    }

    pub async fn tx(&self) -> Result<GraphTx<'_, V>> {
        Ok(GraphTx {
            tx:      self.pool.begin().await?,
            phantom: PhantomData,
        })
    }
}

impl<V: VectorStore> GraphTx<'_, V> {
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
        for ((n, _nq), links) in izip!(plan.neighbors.iter(), plan.n_links) {
            self.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors, lc).await;
    }

    pub async fn get_entry_point(&mut self) -> Option<(V::VectorRef, usize)> {
        sqlx::query(
            "
                SELECT entry_point FROM hawk_graph_entry WHERE id = 0
            ",
        )
        .fetch_optional(self.tx.deref_mut())
        .await
        .expect("Failed to fetch entry point")
        .map(|row: PgRow| {
            let x: Json<EntryPoint<V::VectorRef>> = row.get("entry_point");
            (x.point.clone(), x.layer)
        })
    }

    async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) {
        sqlx::query(
            "
            INSERT INTO hawk_graph_entry (entry_point, id)
            VALUES ($1, 0) ON CONFLICT (id)
            DO UPDATE SET entry_point = EXCLUDED.entry_point
        ",
        )
        .bind(Json(&EntryPoint { point, layer }))
        .execute(self.tx.deref_mut())
        .await
        .expect("Failed to set entry point");
    }

    pub async fn get_links(
        &mut self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> SortedNeighborhoodV<V> {
        sqlx::query(
            "
            SELECT links FROM hawk_graph_links WHERE source_ref = $1 AND layer = $2
        ",
        )
        .bind(Text(base))
        .bind(lc as i32)
        .fetch_optional(self.tx.deref_mut())
        .await
        .expect("Failed to fetch links")
        .map(|row: PgRow| {
            let x: Json<SortedNeighborhoodV<V>> = row.get("links");
            x.as_ref().clone()
        })
        .unwrap_or_else(SortedNeighborhoodV::<V>::new)
    }

    async fn set_links(&mut self, base: V::VectorRef, links: SortedNeighborhoodV<V>, lc: usize) {
        sqlx::query(
            "
            INSERT INTO hawk_graph_links (source_ref, layer, links)
            VALUES ($1, $2, $3) ON CONFLICT (source_ref, layer)
            DO UPDATE SET
            links = EXCLUDED.links
        ",
        )
        .bind(Text(base))
        .bind(lc as i32)
        .bind(Json(&links))
        .execute(self.tx.deref_mut())
        .await
        .expect("Failed to set links");
    }
}

impl<V: VectorStore> GraphTx<'_, V>
where
    V::VectorRef: Send + Unpin + 'static,
    BoxDynError: From<<V::VectorRef as FromStr>::Err>,
    V::DistanceRef: Send + Unpin + 'static,
{
    fn stream_links(&mut self) -> impl Stream<Item = Result<RowLinks<V>>> + '_ {
        sqlx::query_as::<_, RowLinks<V>>(
            "
        SELECT source_ref, links, layer FROM hawk_graph_links
        ",
        )
        .fetch(self.tx.deref_mut())
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

fn sql_switch_schema(schema_name: &str) -> Result<String> {
    sanitize_identifier(schema_name)?;
    Ok(format!(
        "
        CREATE SCHEMA IF NOT EXISTS \"{}\";
        SET search_path TO \"{}\";
        ",
        schema_name, schema_name
    ))
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

pub mod test_utils {
    use super::*;
    use std::{
        env,
        ops::{Deref, DerefMut},
    };
    const DOTENV_TEST: &str = ".env.test";
    const ENV_DB_URL: &str = "SMPC__DATABASE__URL";
    const SCHEMA_PREFIX: &str = "graph_store_test";

    /// A test database. It creates a unique schema for each test. Call
    /// `cleanup` at the end of the test.
    ///
    /// Access the database with `&graph` or `graph.owned()`.
    pub struct TestGraphPg<V: VectorStore> {
        graph:       GraphPg<V>,
        schema_name: String,
    }

    impl<V: VectorStore> TestGraphPg<V> {
        pub async fn new() -> Result<Self> {
            let schema_name = temporary_name();
            let graph = GraphPg::new(&test_db_url()?, &schema_name).await?;
            Ok(TestGraphPg { graph, schema_name })
        }

        pub async fn cleanup(&self) -> Result<()> {
            cleanup(&self.graph.pool, &self.schema_name).await
        }

        pub fn owned(&self) -> GraphPg<V> {
            GraphPg {
                pool:    self.graph.pool.clone(),
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

    fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(env::var(ENV_DB_URL)?)
    }

    fn temporary_name() -> String {
        format!("{}_{}", SCHEMA_PREFIX, rand::random::<u32>())
    }

    async fn cleanup(pool: &sqlx::PgPool, schema_name: &str) -> Result<()> {
        assert!(schema_name.starts_with(SCHEMA_PREFIX));
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
            .execute(pool)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
mod tests {
    use super::{test_utils::TestGraphPg, *};
    use crate::{
        hawkers::plaintext_store::PlaintextStore,
        hnsw::{GraphMem, HnswSearcher, SortedNeighborhood},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_db() {
        let mut graph = TestGraphPg::<PlaintextStore>::new().await.unwrap();
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

        let ep = tx.get_entry_point().await;
        assert!(ep.is_none());

        let ep2 = EntryPoint {
            point: vectors[0],
            layer: ep.map(|e| e.1).unwrap_or_default() + 1,
        };

        tx.set_entry_point(ep2.point, ep2.layer).await;

        let (point3, layer3) = tx.get_entry_point().await.unwrap();
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

            tx.set_links(vectors[i], links.clone(), 0).await;

            let links2 = tx.get_links(&vectors[i], 0).await;
            assert_eq!(*links, *links2);
        }

        tx.tx.commit().await.unwrap();
        graph.cleanup().await.unwrap();
    }

    #[tokio::test]
    async fn test_hnsw_db() {
        let mut graph_pg = TestGraphPg::<PlaintextStore>::new().await.unwrap();
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
            tx.insert_apply(plan).await;
        }

        let graph_mem2 = tx.load_to_mem().await.unwrap();
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
