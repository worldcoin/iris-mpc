use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::{
    stream::{self},
    Stream,
};
use iris_mpc_common::config::Config;
use sqlx::{migrate::Migrator, postgres::PgPoolOptions, Executor, PgPool, Postgres, Transaction};
use std::{ops::DerefMut, pin::Pin};

const APP_NAME: &str = "SMPC";
const MAX_CONNECTIONS: u32 = 100;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

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

#[derive(sqlx::FromRow, Debug, Default, PartialEq, Eq)]
pub struct StoredIris {
    #[allow(dead_code)]
    id:         i64, // BIGSERIAL
    left_code:  Vec<u8>, // BYTEA
    left_mask:  Vec<u8>, // BYTEA
    right_code: Vec<u8>, // BYTEA
    right_mask: Vec<u8>, // BYTEA
}

impl StoredIris {
    /// The index which is contiguous and starts from 0.
    pub fn index(&self) -> usize {
        self.id as usize
    }

    pub fn left_code(&self) -> &[u16] {
        cast_u8_to_u16(&self.left_code)
    }

    pub fn left_mask(&self) -> &[u16] {
        cast_u8_to_u16(&self.left_mask)
    }

    pub fn right_code(&self) -> &[u16] {
        cast_u8_to_u16(&self.right_code)
    }

    pub fn right_mask(&self) -> &[u16] {
        cast_u8_to_u16(&self.right_mask)
    }
    pub fn id(&self) -> i64 {
        self.id
    }
}

#[derive(Clone)]
pub struct StoredIrisRef<'a> {
    pub left_code:  &'a [u16],
    pub left_mask:  &'a [u16],
    pub right_code: &'a [u16],
    pub right_mask: &'a [u16],
}

#[derive(sqlx::FromRow, Debug, Default)]
struct StoredState {
    request_id: String,
}

#[derive(Clone)]
pub struct Store {
    pool: PgPool,
}

impl Store {
    /// Connect to a database based on Config URL, environment, and party_id.
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

        Ok(Store { pool })
    }

    pub async fn tx(&self) -> Result<Transaction<'_, Postgres>> {
        Ok(self.pool.begin().await?)
    }

    pub async fn count_irises(&self) -> Result<usize> {
        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM irises")
            .fetch_one(&self.pool)
            .await?;
        Ok(count.0 as usize)
    }

    /// Stream irises in order.
    pub async fn stream_irises(&self) -> impl Stream<Item = Result<StoredIris, sqlx::Error>> + '_ {
        sqlx::query_as::<_, StoredIris>("SELECT * FROM irises ORDER BY id").fetch(&self.pool)
    }

    /// Stream irises in parallel, without a particular order.
    pub async fn stream_irises_par(
        &self,
        partitions: usize,
    ) -> impl Stream<Item = Result<StoredIris, sqlx::Error>> + '_ {
        let count = self.count_irises().await.expect("Failed count_irises");
        let partition_size = count.div_ceil(partitions).max(1);

        let mut partition_streams = Vec::new();
        for i in 0..partitions {
            let start_id = partition_size * i;
            let end_id = start_id + partition_size - 1;

            let partition_stream =
                sqlx::query_as::<_, StoredIris>("SELECT * FROM irises WHERE id BETWEEN $1 AND $2")
                    .bind(start_id as i64)
                    .bind(end_id as i64)
                    .fetch(&self.pool);

            partition_streams.push(Box::pin(partition_stream)
                as Pin<Box<dyn Stream<Item = Result<StoredIris, sqlx::Error>> + Send>>);
        }

        stream::select_all(partition_streams)
    }

    pub async fn insert_irises(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        codes_and_masks: &[StoredIrisRef<'_>],
    ) -> Result<()> {
        if codes_and_masks.is_empty() {
            return Ok(());
        }
        let mut query = sqlx::QueryBuilder::new(
            "INSERT INTO irises (left_code, left_mask, right_code, right_mask)",
        );
        query.push_values(codes_and_masks, |mut query, iris| {
            query.push_bind(cast_slice::<u16, u8>(iris.left_code));
            query.push_bind(cast_slice::<u16, u8>(iris.left_mask));
            query.push_bind(cast_slice::<u16, u8>(iris.right_code));
            query.push_bind(cast_slice::<u16, u8>(iris.right_mask));
        });

        query.build().execute(tx.deref_mut()).await?;
        Ok(())
    }

    pub async fn insert_or_update_left_iris(
        &self,
        id: i64,
        left_code: &[u16],
        left_mask: &[u16],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let query = sqlx::query(
            r#"
INSERT INTO irises (id, left_code, left_mask) 
VALUES ( $1, $2, $3 ) 
ON CONFLICT (id)
DO UPDATE SET left_code = EXCLUDED.left_code, left_mask = EXCLUDED.left_mask;
"#,
        )
        .bind(id)
        .bind(cast_slice::<u16, u8>(left_code))
        .bind(cast_slice::<u16, u8>(left_mask));

        query.execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn insert_or_update_right_iris(
        &self,
        id: i64,
        right_code: &[u16],
        right_mask: &[u16],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let query = sqlx::query(
            r#"
INSERT INTO irises (id, right_code, right_mask) 
VALUES ( $1, $2, $3 ) 
ON CONFLICT (id)
DO UPDATE SET right_code = EXCLUDED.right_code, right_mask = EXCLUDED.right_mask;
"#,
        )
        .bind(id)
        .bind(cast_slice::<u16, u8>(right_code))
        .bind(cast_slice::<u16, u8>(right_mask));

        query.execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn rollback(&self, db_len: usize) -> Result<()> {
        sqlx::query("DELETE FROM irises WHERE id >= $1")
            .bind(db_len as i64)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn insert_results(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        result_events: &[String],
    ) -> Result<()> {
        if result_events.is_empty() {
            return Ok(());
        }
        let mut query = sqlx::QueryBuilder::new("INSERT INTO results (result_event)");
        query.push_values(result_events, |mut query, result| {
            query.push_bind(result);
        });

        query.build().execute(tx.deref_mut()).await?;
        Ok(())
    }

    pub async fn last_results(&self, count: usize) -> Result<Vec<String>> {
        let mut result_events: Vec<String> =
            sqlx::query_scalar("SELECT result_event FROM results ORDER BY id DESC LIMIT $1")
                .bind(count as i64)
                .fetch_all(&self.pool)
                .await?;
        result_events.reverse();
        Ok(result_events)
    }

    pub async fn mark_requests_deleted(&self, request_ids: &[String]) -> Result<()> {
        if request_ids.is_empty() {
            return Ok(());
        }
        // Insert request_ids that are deleted from the queue.
        let mut query = sqlx::QueryBuilder::new("INSERT INTO sync (request_id)");
        query.push_values(request_ids, |mut query, request_id| {
            query.push_bind(request_id);
        });
        query.build().execute(&self.pool).await?;
        Ok(())
    }

    pub async fn last_deleted_requests(&self, count: usize) -> Result<Vec<String>> {
        let rows = sqlx::query_as::<_, StoredState>("SELECT * FROM sync ORDER BY id DESC LIMIT $1")
            .bind(count as i64)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.into_iter().rev().map(|r| r.request_id).collect())
    }
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

fn cast_u8_to_u16(s: &[u8]) -> &[u16] {
    if s.is_empty() {
        &[] // A literal empty &[u8] may be unaligned.
    } else {
        cast_slice(s)
    }
}

#[cfg(test)]
mod tests {
    const DOTENV_TEST: &str = ".env.test";

    use super::*;
    use futures::TryStreamExt;
    use iris_mpc_common::helpers::smpc_request::ResultEvent;

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_store() -> Result<()> {
        // Create a unique schema for this test.
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let got: Vec<StoredIris> = store.stream_irises_par(2).await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let codes_and_masks = &[
            StoredIrisRef {
                left_code:  &[1, 2, 3, 4],
                left_mask:  &[5, 6, 7, 8],
                right_code: &[9, 10, 11, 12],
                right_mask: &[13, 14, 15, 16],
            },
            StoredIrisRef {
                left_code:  &[1117, 18, 19, 20],
                left_mask:  &[21, 1122, 23, 24],
                right_code: &[25, 26, 1127, 28],
                right_mask: &[29, 30, 31, 1132],
            },
            StoredIrisRef {
                left_code:  &[17, 18, 19, 20],
                left_mask:  &[21, 22, 23, 24],
                // Empty is allowed until stereo is implemented.
                right_code: &[],
                right_mask: &[],
            },
        ];
        let mut tx = store.tx().await?;
        store.insert_irises(&mut tx, &codes_and_masks[0..2]).await?;
        store.insert_irises(&mut tx, &codes_and_masks[2..3]).await?;
        tx.commit().await?;

        let got_len = store.count_irises().await?;
        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;

        let mut got_par: Vec<StoredIris> = store.stream_irises_par(2).await.try_collect().await?;
        got_par.sort_by_key(|iris| iris.id);
        assert_eq!(got, got_par);

        assert_eq!(got_len, 3);
        assert_eq!(got.len(), 3);
        for i in 0..3 {
            assert_eq!(got[i].id, i as i64);
            assert_eq!(got[i].left_code(), codes_and_masks[i].left_code);
            assert_eq!(got[i].left_mask(), codes_and_masks[i].left_mask);
            assert_eq!(got[i].right_code(), codes_and_masks[i].right_code);
            assert_eq!(got[i].right_mask(), codes_and_masks[i].right_mask);
        }

        // Clean up on success.
        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_empty_insert() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &[]).await?;
        store.insert_irises(&mut tx, &[]).await?;
        tx.commit().await?;
        store.mark_requests_deleted(&[]).await?;

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_insert_many() -> Result<()> {
        let count = 1 << 13;

        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let iris = StoredIrisRef {
            left_code:  &[123_u16; 12800],
            left_mask:  &[456_u16; 12800],
            right_code: &[789_u16; 12800],
            right_mask: &[101_u16; 12800],
        };
        let codes_and_masks = vec![iris; count];

        let result_event = serde_json::to_string(&ResultEvent::new(
            0,
            Some(1_000_000_000),
            false,
            "A".repeat(64),
            None,
        ))?;
        let result_events = vec![result_event; count];

        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &result_events).await?;
        store.insert_irises(&mut tx, &codes_and_masks).await?;
        tx.commit().await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), count);
        assert_contiguous_id(&got);

        // Compare with the parallel version with several edge-cases.
        for parallelism in [1, 5, MAX_CONNECTIONS as usize + 1] {
            let mut got_par: Vec<StoredIris> = store
                .stream_irises_par(parallelism)
                .await
                .try_collect()
                .await?;
            got_par.sort_by_key(|iris| iris.id);
            assert_eq!(got, got_par);
        }

        let got = store.last_results(count).await?;
        assert_eq!(got, result_events);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_rollback() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let iris = StoredIrisRef {
            left_code:  &[123_u16; 12800],
            left_mask:  &[456_u16; 12800],
            right_code: &[789_u16; 12800],
            right_mask: &[101_u16; 12800],
        };

        let mut tx = store.tx().await?;
        store.insert_irises(&mut tx, &vec![iris; 10]).await?;
        tx.commit().await?;
        store.rollback(5).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 5);
        assert_contiguous_id(&got);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_results() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let result_events = vec!["event1".to_string(), "event2".to_string()];
        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &result_events).await?;
        store.insert_results(&mut tx, &result_events).await?;
        tx.commit().await?;

        let got = store.last_results(2).await?;
        assert_eq!(got, vec!["event1", "event2"]);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_mark_requests_deleted() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        assert_eq!(store.last_deleted_requests(2).await?.len(), 0);

        for i in 0..2 {
            let request_ids = (0..2)
                .map(|j| format!("test_{}_{}", i, j))
                .collect::<Vec<_>>();
            store.mark_requests_deleted(&request_ids).await?;

            let got = store.last_deleted_requests(2).await?;
            assert_eq!(got, request_ids);
        }

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn test_insert_left_right() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        for i in 0..10u16 {
            if i % 2 == 0 {
                store
                    .insert_or_update_left_iris(
                        i as i64,
                        &[i * 100 + 1, i * 100 + 2, i * 100 + 3, i * 100 + 4],
                        &[i * 100 + 5, i * 100 + 6, i * 100 + 7, i * 100 + 8],
                    )
                    .await?;
                store
                    .insert_or_update_right_iris(
                        i as i64,
                        &[i * 100 + 9, i * 100 + 10, i * 100 + 11, i * 100 + 12],
                        &[i * 100 + 13, i * 100 + 14, i * 100 + 15, i * 100 + 16],
                    )
                    .await?;
            } else {
                store
                    .insert_or_update_right_iris(
                        i as i64,
                        &[i * 100 + 9, i * 100 + 10, i * 100 + 11, i * 100 + 12],
                        &[i * 100 + 13, i * 100 + 14, i * 100 + 15, i * 100 + 16],
                    )
                    .await?;
                store
                    .insert_or_update_left_iris(
                        i as i64,
                        &[i * 100 + 1, i * 100 + 2, i * 100 + 3, i * 100 + 4],
                        &[i * 100 + 5, i * 100 + 6, i * 100 + 7, i * 100 + 8],
                    )
                    .await?;
            }
        }

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;

        for i in 0..10u16 {
            assert_eq!(got[i as usize].id, i as i64);
            assert_eq!(got[i as usize].left_code(), &[
                i * 100 + 1,
                i * 100 + 2,
                i * 100 + 3,
                i * 100 + 4
            ]);
            assert_eq!(got[i as usize].left_mask(), &[
                i * 100 + 5,
                i * 100 + 6,
                i * 100 + 7,
                i * 100 + 8
            ]);
            assert_eq!(got[i as usize].right_code(), &[
                i * 100 + 9,
                i * 100 + 10,
                i * 100 + 11,
                i * 100 + 12
            ]);
            assert_eq!(got[i as usize].right_mask(), &[
                i * 100 + 13,
                i * 100 + 14,
                i * 100 + 15,
                i * 100 + 16
            ]);
        }

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(Config::load_config(APP_NAME)?
            .database
            .ok_or(eyre!("Missing database config"))?
            .url)
    }

    fn temporary_name() -> String {
        format!("smpc_test{}_0", rand::random::<u32>())
    }

    fn assert_contiguous_id(vec: &[StoredIris]) {
        assert!(
            vec.iter().enumerate().all(|(i, row)| row.id == i as i64),
            "IDs must be contiguous and in order"
        );
    }

    async fn cleanup(store: &Store, schema_name: &str) -> Result<()> {
        assert!(schema_name.starts_with("smpc_test"));
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
            .execute(&store.pool)
            .await?;
        Ok(())
    }
}
