#![feature(int_roundings)]

mod s3_importer;

use crate::s3_importer::S3StoredIris;
use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::{
    stream::{self},
    Stream, StreamExt, TryStreamExt,
};
use iris_mpc_common::{
    config::Config,
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
pub use s3_importer::{fetch_and_parse_chunks, last_snapshot_timestamp, ObjectStore, S3Store};
use sqlx::{
    migrate::Migrator, postgres::PgPoolOptions, Executor, PgPool, Postgres, Row, Transaction,
};
use std::ops::DerefMut;

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

/// The unified type that can hold either DB or S3 variants.
pub enum StoredIris {
    // DB stores the shares in their original form
    DB(DbStoredIris),
    // S3 stores the shares in even-odd ordered form (needed by limbs) to make the loading faster
    S3(S3StoredIris),
}

impl StoredIris {
    /// Returns the `id` from either variant.
    pub fn index(&self) -> usize {
        match self {
            StoredIris::DB(db) => db.index(),
            StoredIris::S3(s3) => s3.index(),
        }
    }
}

#[derive(sqlx::FromRow, Debug, Default, PartialEq, Eq)]
pub struct DbStoredIris {
    #[allow(dead_code)]
    id:         i64, // BIGSERIAL
    left_code:  Vec<u8>, // BYTEA
    left_mask:  Vec<u8>, // BYTEA
    right_code: Vec<u8>, // BYTEA
    right_mask: Vec<u8>, // BYTEA
}

impl DbStoredIris {
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
    pub id:         i64,
    pub left_code:  &'a [u16],
    pub left_mask:  &'a [u16],
    pub right_code: &'a [u16],
    pub right_mask: &'a [u16],
}

#[derive(sqlx::FromRow, Debug, Default)]
struct StoredState {
    request_id: String,
}

#[derive(Clone, Debug)]
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
        tracing::info!("Connecting to V2 database with, schema: {}", schema_name);
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
    pub async fn stream_irises(
        &self,
    ) -> impl Stream<Item = Result<DbStoredIris, sqlx::Error>> + '_ {
        sqlx::query_as::<_, DbStoredIris>("SELECT * FROM irises WHERE id >= 1 ORDER BY id")
            .fetch(&self.pool)
    }

    pub fn stream_irises_in_range(
        &self,
        id_range: std::ops::Range<u64>,
    ) -> impl Stream<Item = sqlx::Result<DbStoredIris>> + '_ {
        sqlx::query_as(
            r#"
            SELECT *
            FROM irises
            WHERE id >= $1 AND id < $2
            ORDER BY id ASC
            "#,
        )
        .bind(i64::try_from(id_range.start).expect("id fits into i64"))
        .bind(i64::try_from(id_range.end).expect("id fits into i64"))
        .fetch(&self.pool)
    }
    /// Stream irises in parallel, without a particular order.
    pub async fn stream_irises_par(
        &self,
        min_last_modified_at: Option<i64>,
        partitions: usize,
    ) -> impl Stream<Item = eyre::Result<StoredIris>> + '_ {
        let count = self.count_irises().await.expect("Failed count_irises");
        let partition_size = count.div_ceil(partitions).max(1);

        let mut partition_streams = Vec::new();
        for i in 0..partitions {
            // we start from ID 1
            let start_id = 1 + partition_size * i;
            let end_id = start_id + partition_size - 1;

            // This base query yields `DbStoredIris`
            let base_stream = match min_last_modified_at {
                Some(min_last_modified_at) => sqlx::query_as::<_, DbStoredIris>(
                    "SELECT id, left_code, left_mask, right_code, right_mask FROM irises WHERE id \
                     BETWEEN $1 AND $2 AND last_modified_at >= $3",
                )
                .bind(start_id as i64)
                .bind(end_id as i64)
                .bind(min_last_modified_at)
                .fetch(&self.pool),
                None => sqlx::query_as::<_, DbStoredIris>(
                    "SELECT id, left_code, left_mask, right_code, right_mask FROM irises WHERE id \
                     BETWEEN $1 AND $2",
                )
                .bind(start_id as i64)
                .bind(end_id as i64)
                .fetch(&self.pool),
            };

            // Convert `Stream<Item = Result<DbStoredIris, sqlx::Error>>`
            //  -> `Stream<Item = eyre::Result<DbStoredIris>>` using map_err,
            //  -> then map_ok(StoredIris::Db) to unify the output type:
            let partition_stream = base_stream
                .map_err(Into::into) // `sqlx::Error` -> `eyre::Error`
                .map_ok(StoredIris::DB) // `DbStoredIris` -> `StoredIris::Db(...)`
                .boxed();

            partition_streams.push(partition_stream);
        }

        // `select_all` requires that all streams have the same Item type:
        // which is `Result<StoredIris, eyre::Error>` now.
        stream::select_all(partition_streams)
    }

    pub async fn insert_irises(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        codes_and_masks: &[StoredIrisRef<'_>],
    ) -> Result<Vec<i64>> {
        if codes_and_masks.is_empty() {
            return Ok(vec![]);
        }
        let mut query = sqlx::QueryBuilder::new(
            "INSERT INTO irises (id, left_code, left_mask, right_code, right_mask)",
        );
        query.push_values(codes_and_masks, |mut query, iris| {
            query.push_bind(iris.id);
            query.push_bind(cast_slice::<u16, u8>(iris.left_code));
            query.push_bind(cast_slice::<u16, u8>(iris.left_mask));
            query.push_bind(cast_slice::<u16, u8>(iris.right_code));
            query.push_bind(cast_slice::<u16, u8>(iris.right_mask));
        });

        query.push(" RETURNING id");

        let ids = query
            .build()
            .fetch_all(tx.deref_mut())
            .await?
            .iter()
            .map(|row| row.get::<i64, _>("id"))
            .collect::<Vec<_>>();

        Ok(ids)
    }

    pub async fn insert_irises_overriding(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        codes_and_masks: &[StoredIrisRef<'_>],
    ) -> Result<()> {
        if codes_and_masks.is_empty() {
            return Ok(());
        }
        let mut query = sqlx::QueryBuilder::new(
            "INSERT INTO irises (id, left_code, left_mask, right_code, right_mask)",
        );
        query.push_values(codes_and_masks, |mut query, iris| {
            query.push_bind(iris.id);
            query.push_bind(cast_slice::<u16, u8>(iris.left_code));
            query.push_bind(cast_slice::<u16, u8>(iris.left_mask));
            query.push_bind(cast_slice::<u16, u8>(iris.right_code));
            query.push_bind(cast_slice::<u16, u8>(iris.right_mask));
        });
        query.push(
            r#"
ON CONFLICT (id)
DO UPDATE SET left_code = EXCLUDED.left_code, left_mask = EXCLUDED.left_mask, right_code = EXCLUDED.right_code, right_mask = EXCLUDED.right_mask;
"#,
        );

        query.build().execute(tx.deref_mut()).await?;

        Ok(())
    }

    /// Update existing iris with given shares.
    pub async fn update_iris(
        &self,
        id: i64,
        left_iris_share: &GaloisRingIrisCodeShare,
        left_mask_share: &GaloisRingTrimmedMaskCodeShare,
        right_iris_share: &GaloisRingIrisCodeShare,
        right_mask_share: &GaloisRingTrimmedMaskCodeShare,
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let query = sqlx::query(
            r#"
UPDATE irises SET (left_code, left_mask, right_code, right_mask) = ($2, $3, $4, $5)
WHERE id = $1;
"#,
        )
        .bind(id)
        .bind(cast_slice::<u16, u8>(&left_iris_share.coefs[..]))
        .bind(cast_slice::<u16, u8>(&left_mask_share.coefs[..]))
        .bind(cast_slice::<u16, u8>(&right_iris_share.coefs[..]))
        .bind(cast_slice::<u16, u8>(&right_mask_share.coefs[..]));

        query.execute(&mut *tx).await?;
        tx.commit().await?;
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
        let mut tx = self.pool.begin().await?;

        sqlx::query("DELETE FROM irises WHERE id > $1")
            .bind(db_len as i64)
            .execute(&mut *tx)
            .await?;

        tx.commit().await?;
        Ok(())
    }

    pub async fn get_max_serial_id(&self) -> Result<usize> {
        let id: (i64,) = sqlx::query_as("SELECT MAX(id) FROM irises")
            .fetch_one(&self.pool)
            .await?;
        Ok(id.0 as usize)
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

    /// Initialize the database with random shares and masks. Cleans up the db
    /// before inserting new generated irises.
    pub async fn init_db_with_random_shares(
        &self,
        rng_seed: u64,
        party_id: usize,
        db_size: usize,
        clear_db_before_init: bool,
    ) -> Result<()> {
        let mut rng = StdRng::seed_from_u64(rng_seed);

        if clear_db_before_init {
            tracing::info!("Cleaning up the db before initializing irises");
            // Cleaning up the db before inserting newly generated irises
            self.rollback(0).await?;
        }

        let mut tx = self.tx().await?;

        tracing::info!(
            "DB size before initialization: {}",
            self.count_irises().await?
        );
        for i in 0..db_size {
            if (i % 1000) == 0 {
                tracing::info!("Initializing iris db: Generated {} entries", i);
            }

            let mut rng = StdRng::from_seed(rng.gen());
            let iris = IrisCode::random_rng(&mut rng);

            let share = GaloisRingIrisCodeShare::encode_iris_code(
                &iris.code,
                &iris.mask,
                &mut StdRng::seed_from_u64(rng_seed),
            )[party_id]
                .clone();

            let mask: GaloisRingTrimmedMaskCodeShare = GaloisRingIrisCodeShare::encode_mask_code(
                &iris.mask,
                &mut StdRng::seed_from_u64(rng_seed),
            )[party_id]
                .clone()
                .into();

            // inserting shares and masks in the db. Reusing the same share and mask for
            // left and right
            self.insert_irises(&mut tx, &[StoredIrisRef {
                id:         (i + 1) as i64,
                left_code:  &share.coefs,
                left_mask:  &mask.coefs,
                right_code: &share.coefs,
                right_mask: &mask.coefs,
            }])
            .await?;

            if (i % 1000) == 0 {
                tx.commit().await?;
                tx = self.tx().await?;
            }
        }
        tracing::info!("Completed initialization of iris db, committing...");
        tx.commit().await?;
        tracing::info!("Committed");

        tracing::info!(
            "Initialized iris db with {} entries",
            self.count_irises().await?
        );
        Ok(())
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
#[cfg(feature = "db_dependent")]
mod tests {
    const DOTENV_TEST: &str = ".env.test";

    use super::*;
    use futures::TryStreamExt;
    use iris_mpc_common::helpers::smpc_response::UniquenessResult;

    #[tokio::test]
    async fn test_store() -> Result<()> {
        // Create a unique schema for this test.
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let got: Vec<DbStoredIris> = store
            .stream_irises_par(Some(0), 2)
            .await
            .map_ok(|stored_iris| match stored_iris {
                StoredIris::DB(db_iris) => db_iris,
                StoredIris::S3(_) => panic!("Unexpected S3 variant in this test!"),
            })
            .try_collect()
            .await?;
        assert_eq!(got.len(), 0);

        let codes_and_masks = &[
            StoredIrisRef {
                id:         1,
                left_code:  &[1, 2, 3, 4],
                left_mask:  &[5, 6, 7, 8],
                right_code: &[9, 10, 11, 12],
                right_mask: &[13, 14, 15, 16],
            },
            StoredIrisRef {
                id:         2,
                left_code:  &[1117, 18, 19, 20],
                left_mask:  &[21, 1122, 23, 24],
                right_code: &[25, 26, 1127, 28],
                right_mask: &[29, 30, 31, 1132],
            },
            StoredIrisRef {
                id:         3,
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
        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;

        let mut got_par: Vec<DbStoredIris> = store
            .stream_irises_par(Some(0), 2)
            .await
            .map_ok(|stored_iris| match stored_iris {
                StoredIris::DB(db_iris) => db_iris,
                StoredIris::S3(_) => panic!("Unexpected S3 variant in this test!"),
            })
            .try_collect()
            .await?;
        got_par.sort_by_key(|iris| iris.id);
        assert_eq!(got, got_par);

        assert_eq!(got_len, 3);
        assert_eq!(got.len(), 3);
        for i in 0..3 {
            assert_eq!(got[i].id, (i + 1) as i64);
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
    async fn test_insert_many() -> Result<()> {
        let count: usize = 1 << 3;

        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let mut codes_and_masks = vec![];

        for i in 0..count {
            let iris = StoredIrisRef {
                id:         (i + 1) as i64,
                left_code:  &[123_u16; 12800],
                left_mask:  &[456_u16; 12800],
                right_code: &[789_u16; 12800],
                right_mask: &[101_u16; 12800],
            };
            codes_and_masks.push(iris);
        }

        let result_event = serde_json::to_string(&UniquenessResult::new(
            0,
            Some(1_000_000_000),
            false,
            "A".repeat(64),
            None,
            None,
            None,
            None,
        ))?;
        let result_events = vec![result_event; count];

        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &result_events).await?;
        store.insert_irises(&mut tx, &codes_and_masks).await?;
        tx.commit().await?;

        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), count);
        assert_contiguous_id(&got);

        // Compare with the parallel version with several edge-cases.
        for parallelism in [1, 5, MAX_CONNECTIONS as usize + 1] {
            let mut got_par: Vec<DbStoredIris> = store
                .stream_irises_par(Some(0), parallelism)
                .await
                .map_ok(|stored_iris| match stored_iris {
                    StoredIris::DB(db_iris) => db_iris,
                    StoredIris::S3(_) => panic!("Unexpected S3 variant in this test!"),
                })
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
    async fn test_init_db_with_random_shares() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let expected_generated_irises_num = 10;
        store
            .init_db_with_random_shares(0, 0, expected_generated_irises_num, true)
            .await?;

        let generated_irises_count = store.count_irises().await?;
        assert_eq!(generated_irises_count, expected_generated_irises_num);

        cleanup(&store, &schema_name).await
    }

    #[tokio::test]
    async fn test_rollback() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let mut irises = vec![];
        for i in 0..10 {
            let iris = StoredIrisRef {
                id:         (i + 1) as i64,
                left_code:  &[123_u16; 12800],
                left_mask:  &[456_u16; 12800],
                right_code: &[789_u16; 12800],
                right_mask: &[101_u16; 12800],
            };
            irises.push(iris);
        }

        let mut tx = store.tx().await?;
        store.insert_irises(&mut tx, &irises).await?;
        tx.commit().await?;
        store.rollback(5).await?;

        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 5);
        assert_contiguous_id(&got);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
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
    async fn test_insert_left_right() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        for i in 0..10u16 {
            if i % 2 == 0 {
                store
                    .insert_or_update_left_iris(
                        (i + 1) as i64,
                        &[i * 100 + 1, i * 100 + 2, i * 100 + 3, i * 100 + 4],
                        &[i * 100 + 5, i * 100 + 6, i * 100 + 7, i * 100 + 8],
                    )
                    .await?;
                store
                    .insert_or_update_right_iris(
                        (i + 1) as i64,
                        &[i * 100 + 9, i * 100 + 10, i * 100 + 11, i * 100 + 12],
                        &[i * 100 + 13, i * 100 + 14, i * 100 + 15, i * 100 + 16],
                    )
                    .await?;
            } else {
                store
                    .insert_or_update_right_iris(
                        (i + 1) as i64,
                        &[i * 100 + 9, i * 100 + 10, i * 100 + 11, i * 100 + 12],
                        &[i * 100 + 13, i * 100 + 14, i * 100 + 15, i * 100 + 16],
                    )
                    .await?;
                store
                    .insert_or_update_left_iris(
                        (i + 1) as i64,
                        &[i * 100 + 1, i * 100 + 2, i * 100 + 3, i * 100 + 4],
                        &[i * 100 + 5, i * 100 + 6, i * 100 + 7, i * 100 + 8],
                    )
                    .await?;
            }
        }

        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;

        for i in 0..10u16 {
            assert_eq!(got[i as usize].id, (i + 1) as i64);
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

    #[tokio::test]
    async fn test_update_iris() -> Result<()> {
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        // insert two irises into db
        let iris1 = StoredIrisRef {
            id:         1,
            left_code:  &[123_u16; 12800],
            left_mask:  &[456_u16; 6400],
            right_code: &[789_u16; 12800],
            right_mask: &[101_u16; 6400],
        };
        let mut iris2 = iris1.clone();
        iris2.id = 2;

        let mut tx = store.tx().await?;
        store
            .insert_irises(&mut tx, &[iris1, iris2.clone()])
            .await?;
        tx.commit().await?;

        // update iris with id 1 in db
        let updated_left_code = GaloisRingIrisCodeShare {
            id:    1,
            coefs: [666_u16; 12800],
        };
        let updated_left_mask = GaloisRingTrimmedMaskCodeShare {
            id:    1,
            coefs: [777_u16; 6400],
        };
        let updated_right_code = GaloisRingIrisCodeShare {
            id:    1,
            coefs: [888_u16; 12800],
        };
        let updated_right_mask = GaloisRingTrimmedMaskCodeShare {
            id:    1,
            coefs: [999_u16; 6400],
        };
        store
            .update_iris(
                1,
                &updated_left_code,
                &updated_left_mask,
                &updated_right_code,
                &updated_right_mask,
            )
            .await?;

        // assert iris updated in db with new values
        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 2);
        assert_eq!(cast_u8_to_u16(&got[0].left_code), updated_left_code.coefs);
        assert_eq!(cast_u8_to_u16(&got[0].left_mask), updated_left_mask.coefs);
        assert_eq!(cast_u8_to_u16(&got[0].right_code), updated_right_code.coefs);
        assert_eq!(cast_u8_to_u16(&got[0].right_mask), updated_right_mask.coefs);

        // assert the other iris in db is not updated
        assert_eq!(cast_u8_to_u16(&got[1].left_code), iris2.left_code);
        assert_eq!(cast_u8_to_u16(&got[1].left_mask), iris2.left_mask);
        assert_eq!(cast_u8_to_u16(&got[1].right_code), iris2.right_code);
        assert_eq!(cast_u8_to_u16(&got[1].right_mask), iris2.right_mask);

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

    fn assert_contiguous_id(vec: &[DbStoredIris]) {
        assert!(
            vec.iter()
                .enumerate()
                .all(|(i, row)| row.id == (i + 1) as i64),
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
