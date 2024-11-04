use super::plaintext_store::{PlaintextPoint, PlaintextStore, PointId};
use eyre::{eyre, Result};
use futures::stream::TryStreamExt;
use hawk_pack::{DbStore, VectorStore};
use sqlx::{
    migrate::Migrator,
    postgres::{PgPoolOptions, PgRow},
    Executor, PgPool, Row,
};
use std::{collections::BTreeMap, path};
use tokio::io::AsyncWriteExt;

const MAX_CONNECTIONS: u32 = 5;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

#[derive(Debug, Clone)]
pub struct PlaintextStoreDb {
    cache:       PlaintextStore,
    schema_name: String,
    pool:        sqlx::PgPool,
}

impl VectorStore for PlaintextStoreDb {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.
    type Data = PlaintextPoint;

    fn prepare_query(&mut self, raw_query: PlaintextPoint) -> PointId {
        self.cache.prepare_query(raw_query)
    }

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        let point = self.get_point(*query).await.unwrap();

        sqlx::query(
            "
            INSERT INTO hawk_vectors (id, point)
            VALUES ($1, $2)
        ",
        )
        .bind(query.val() as i32)
        .bind(sqlx::types::Json(point))
        .execute(&self.pool)
        .await
        .expect(&format!(
            "Failed to insert query {} into vector store",
            query.val()
        ));

        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        let x = &self.get_point(distance.0).await.unwrap();
        let y = &self.get_point(distance.1).await.unwrap();
        x.is_close(y)
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let (d2t1, d1t2) = self.distance_computation(distance1, distance2).await;
        (d2t1 - d1t2) < 0
    }
}

impl DbStore for PlaintextStoreDb {
    async fn new(url: &str, schema_name: &str) -> Result<Self> {
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

        Ok(PlaintextStoreDb {
            cache: PlaintextStore {
                points: BTreeMap::new(),
            },
            schema_name: schema_name.to_owned(),
            pool,
        })
    }

    fn pool(&self) -> &PgPool {
        &self.pool
    }

    fn schema_name(&self) -> String {
        self.schema_name.to_string()
    }

    async fn copy_out(&self) -> Result<Vec<(String, String)>> {
        let file_name = format!("{}_vectors.csv", self.schema_name.clone());
        self.copy_out_with_filename(file_name).await
    }
}

impl PlaintextStoreDb {
    pub async fn to_plaintext_store(&self) -> PlaintextStore {
        let points = sqlx::query(
            "
                SELECT * FROM hawk_vectors
            ",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap()
        .iter()
        .map(|row| {
            let id: i32 = row.get("id");
            let point: sqlx::types::Json<PlaintextPoint> = row.get("point");
            (PointId(id as usize), point.as_ref().clone())
        })
        .collect();

        PlaintextStore { points }
    }

    pub async fn get_point(&self, point: PointId) -> Option<PlaintextPoint> {
        let mut res = self.cache.points.get(&point).map(|p| p.clone());
        if res.is_none() {
            res = sqlx::query(
                "
                    SELECT point FROM hawk_vectors WHERE id = $1
                ",
            )
            .bind(point.0 as i32)
            .fetch_optional(&self.pool)
            .await
            .expect(&format!("Failed to fetch point {}", point.0))
            .map(|row: PgRow| {
                let x: sqlx::types::Json<PlaintextPoint> = row.get("point");
                x.as_ref().clone()
            });
        }
        res
    }

    pub async fn distance_computation(
        &self,
        distance1: &(PointId, PointId),
        distance2: &(PointId, PointId),
    ) -> (i32, i32) {
        let (x1, y1) = (
            &self.get_point(distance1.0).await.unwrap(),
            &self.get_point(distance1.1).await.unwrap(),
        );
        let (x2, y2) = (
            &self.get_point(distance2.0).await.unwrap(),
            &self.get_point(distance2.1).await.unwrap(),
        );
        let (d1, t1) = x1.compute_distance(y1);
        let (d2, t2) = x2.compute_distance(y2);

        let cross_1 = d2 as i32 * t1 as i32;
        let cross_2 = d1 as i32 * t2 as i32;
        (cross_1, cross_2)
    }

    async fn copy_out_with_filename(&self, file_name: String) -> Result<Vec<(String, String)>> {
        let table_name = "hawk_vectors";

        let path = path::absolute(file_name.clone())?
            .as_os_str()
            .to_str()
            .unwrap()
            .to_owned();

        let mut file = tokio::fs::File::create(path.clone()).await?;
        let mut conn = self.pool.acquire().await?;

        let mut copy_stream = conn
            .copy_out_raw(&format!(
                "COPY {} TO STDOUT (FORMAT CSV, HEADER)",
                table_name
            ))
            .await?;

        while let Some(chunk) = copy_stream.try_next().await? {
            file.write_all(&chunk).await?;
        }

        Ok(vec![(table_name.to_string(), path)])
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkers::plaintext_store::FormattedIris;
    use aes_prng::AesRng;
    use hawk_pack::{graph_store::GraphPg, hnsw_db::HawkSearcher};
    use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
    use rand::SeedableRng;

    struct PlaintextPointReader<R: std::io::BufRead> {
        inner: R,
    }

    impl<R: std::io::BufRead> PlaintextPointReader<R> {
        pub fn new(inner: R) -> Self {
            Self { inner }
        }
    }

    impl<R: std::io::BufRead> Iterator for PlaintextPointReader<R> {
        type Item = Vec<i8>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut buf: [u8; 12800] = [0; 12800];
            self.inner.read_exact(&mut buf).ok()?;

            Some(
                buf.iter()
                    .map(|&byte| match byte {
                        2 => -1,
                        _ => byte as i8,
                    })
                    .collect(),
            )
        }
    }

    fn plaintext_point_from_data(data: Vec<i8>) -> PlaintextPoint {
        let mut mask = IrisCodeArray::ZERO;

        for (bit, &val) in data.iter().enumerate() {
            if val != 0 {
                mask.set_bit(bit, true);
            }
        }
        let iris_code = FormattedIris { data, mask };
        PlaintextPoint {
            data:          iris_code,
            is_persistent: false,
        }
    }

    #[tokio::test]
    async fn hawk_searcher_from_db() {
        let database_size = 100;
        let schema_name = format!("hnsw_db_{}", database_size.to_string());
        let temporary_name = || format!("{}_{}", schema_name, rand::random::<u32>());
        let hawk_database_url: &str = "postgres://postgres:postgres@localhost/postgres";

        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut graph_store =
            GraphPg::<PlaintextStoreDb>::new(hawk_database_url, &temporary_name())
                .await
                .unwrap();
        let mut vector_store = PlaintextStoreDb::new(hawk_database_url, &temporary_name())
            .await
            .unwrap();
        let plain_searcher = HawkSearcher::default();

        let queries = (0..database_size)
            .map(|_| {
                let raw_query = IrisCode::random_rng(&mut rng);
                vector_store.prepare_query(raw_query.into())
            })
            .collect::<Vec<_>>();

        for query in queries.iter() {
            let neighbors = plain_searcher
                .search_to_insert(&mut vector_store, &mut graph_store, query)
                .await;
            let inserted = vector_store.insert(query).await;
            plain_searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    &mut rng,
                    inserted,
                    neighbors,
                )
                .await;
        }
        let graph_path = graph_store.copy_out().await.unwrap();
        let vectors_path = vector_store.copy_out().await.unwrap();

        graph_store.cleanup().await.unwrap();
        vector_store.cleanup().await.unwrap();

        // Copy in to memory
        {
            let graph_store =
                GraphPg::<PlaintextStoreDb>::new(hawk_database_url, &temporary_name())
                    .await
                    .unwrap();
            let vector_store = PlaintextStoreDb::new(hawk_database_url, &temporary_name())
                .await
                .unwrap();

            graph_store.copy_in(graph_path).await.unwrap();
            let mut graph_mem = graph_store.to_graph_mem().await;

            vector_store.copy_in(vectors_path).await.unwrap();
            let mut vector_mem = vector_store.to_plaintext_store().await;
            vector_store.cleanup().await.unwrap();

            let plain_searcher = HawkSearcher::default();

            for query in queries.iter() {
                let neighbors = plain_searcher
                    .search_to_insert(&mut vector_mem, &mut graph_mem, query)
                    .await;
                assert!(plain_searcher.is_match(&mut vector_mem, &neighbors).await);
            }
            graph_store.cleanup().await.unwrap();
        }
    }

    #[tokio::test]
    async fn checkpoint_from_data() {
        use std::io::BufReader;

        let step_size = 50000;
        let database_size = 1000000;
        let m_values = [128];

        let mut rng = AesRng::seed_from_u64(0_u64);
        let hawk_database_url: &str = "postgres://postgres:postgres@localhost/postgres";

        let mut queries = vec![];
        for chunk in 0..10 {
            let dat_filename = format!("benches/assets/processed_masked_irises_chunk_{}", chunk);
            let dat_path = path::absolute(dat_filename).unwrap();

            let input =
                BufReader::new(std::fs::File::open(dat_path.clone()).expect("Failed to open file"));
            let values: Vec<Vec<i8>> = PlaintextPointReader::new(input).collect();
            let mut values: Vec<_> = values
                .into_iter()
                .map(|data| plaintext_point_from_data(data))
                .collect();
            queries.append(&mut values);
        }

        for m in m_values.iter() {
            let mut prev_checkpoint_name = None;
            for checkpoint in 1..=(database_size / step_size) {
                let checkpoint_time = std::time::Instant::now();
                println!("M: {:?}, checkpoint: {:?}", m, checkpoint * step_size);

                // Copy in vectors
                let mut vector_mem = PlaintextStore {
                    points: queries
                        .iter()
                        .enumerate()
                        .map(|(id, val)| (PointId(id), val.clone()))
                        .collect(),
                };

                println!("vector_mem.points.len(): {:?}", vector_mem.points.len());

                // Copy in graph
                let mut graph_mem = {
                    let graph_store =
                        GraphPg::<PlaintextStoreDb>::new(hawk_database_url, &"hnsw_1M")
                            .await
                            .unwrap();
                    if let Some(prev_checkpoint_name) = prev_checkpoint_name {
                        graph_store.copy_in(prev_checkpoint_name).await.unwrap();
                    }
                    let graph_mem = graph_store.to_graph_mem().await;
                    graph_store.cleanup().await.unwrap();
                    graph_mem
                };

                println!(
                    "graph_mem.layers: {:?}",
                    graph_mem
                        .get_layers()
                        .iter()
                        .map(|layer| layer.get_links_map().iter())
                        .fold(0, |acc, link| { acc + link.len() })
                );

                let hawk_searcher = HawkSearcher::new_with_m(*m);

                let queries = vector_mem.points.clone();
                for (query, _) in queries.range(PointId((checkpoint - 1) * step_size)..) {
                    if query.val() >= 1000 && (query.val() % 1000) == 0 {
                        println!(
                            "{:?}s: Inserting vector {}",
                            checkpoint_time.elapsed().as_secs(),
                            query.val()
                        );
                    }
                    let neighbors = hawk_searcher
                        .search_to_insert(&mut vector_mem, &mut graph_mem, query)
                        .await;
                    hawk_searcher
                        .insert_from_search_results(
                            &mut vector_mem,
                            &mut graph_mem,
                            &mut rng,
                            *query,
                            neighbors,
                        )
                        .await;
                }
                println!(
                    "{:?}s: Done searching and inserting vectors",
                    checkpoint_time.elapsed().as_secs()
                );

                let checkpoint_name = format!(
                    "1M_{}_M{}_checkpoint",
                    (checkpoint * step_size).to_string(),
                    m.to_string()
                );
                prev_checkpoint_name = Some(
                    graph_mem
                        // .copy_out_with_filename(checkpoint_name)
                        .write_to_db(hawk_database_url, &checkpoint_name)
                        .await
                        .unwrap(),
                );
            }
        }
    }
}
