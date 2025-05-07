mod s3_importer;

use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::{
    stream::{self},
    Stream, StreamExt, TryStreamExt,
};
use iris_mpc_common::{
    config::Config,
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::sync::{Modification, ModificationStatus},
    iris_db::iris::IrisCode,
    postgres::PostgresClient,
    vector_id::VectorId,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
pub use s3_importer::{
    fetch_and_parse_chunks, last_snapshot_timestamp, ObjectStore, S3Store, S3StoredIris,
};
use sqlx::{PgPool, Postgres, Row, Transaction};
use std::ops::DerefMut;

/// The unified type that can hold either DB or S3 variants.
pub enum StoredIris {
    // DB stores the shares in their original form
    DB(DbStoredIris),
    // S3 stores the shares in even-odd ordered form (needed by limbs) to make the loading faster
    S3(S3StoredIris),
}

impl StoredIris {
    /// Returns the `serial_id` from either variant.
    pub fn serial_id(&self) -> usize {
        match self {
            StoredIris::DB(db) => db.serial_id(),
            StoredIris::S3(s3) => s3.serial_id(),
        }
    }
}

#[derive(sqlx::FromRow, Debug, Default, PartialEq, Eq)]
pub struct DbStoredIris {
    id: i64,             // BIGSERIAL
    version_id: i16,     // SMALLINT
    left_code: Vec<u8>,  // BYTEA
    left_mask: Vec<u8>,  // BYTEA
    right_code: Vec<u8>, // BYTEA
    right_mask: Vec<u8>, // BYTEA
}

impl DbStoredIris {
    /// The index which is contiguous and starts from 1.
    pub fn serial_id(&self) -> usize {
        self.id as usize
    }

    pub fn version_id(&self) -> i16 {
        self.version_id
    }

    pub fn vector_id(&self) -> VectorId {
        // TODO: Distinguish vector_id from serial_id.
        VectorId::from_serial_id(self.id as u32)
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
    pub id: i64,
    pub left_code: &'a [u16],
    pub left_mask: &'a [u16],
    pub right_code: &'a [u16],
    pub right_mask: &'a [u16],
}

#[derive(sqlx::FromRow, Debug, Default)]
struct StoredState {
    request_id: String,
}

#[derive(sqlx::FromRow, Debug, Default)]
pub struct StoredModification {
    pub id: i64,
    pub serial_id: i64,
    pub request_type: String,
    pub s3_url: Option<String>,
    pub status: String,
    pub persisted: bool,
    pub result_message_body: Option<String>,
}

impl From<StoredModification> for Modification {
    fn from(stored: StoredModification) -> Self {
        Self {
            id: stored.id,
            serial_id: stored.serial_id,
            request_type: stored.request_type,
            s3_url: stored.s3_url,
            status: stored.status,
            persisted: stored.persisted,
            result_message_body: stored.result_message_body,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Store {
    pub pool: PgPool,
    pub schema_name: String,
}

impl Store {
    pub async fn new(postgres_client: &PostgresClient) -> Result<Self> {
        tracing::info!(
            "Created and iris-mpc-store with schema: {}",
            postgres_client.schema_name
        );

        postgres_client.migrate().await;

        Ok(Store {
            pool: postgres_client.pool.clone(),
            schema_name: postgres_client.schema_name.to_string(),
        })
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

    /// Fetches an ordered vector of rows from Iris table matched by a set of serial identifiers.
    ///
    /// # Arguments
    ///
    /// * `identifiers` - Serial identifiers of Irises to be fetched.
    ///
    /// # Returns
    ///
    /// An ordered vector of `DbStoredIris` instances.
    ///
    pub async fn fetch_iris_batch(
        &self,
        identifiers: Vec<u64>,
    ) -> sqlx::Result<Vec<DbStoredIris>, sqlx::Error> {
        // TODO: define max batch size constant.
        assert!(
            !identifiers.is_empty() && identifiers.len() <= 64,
            "Invalid identifier set"
        );

        tracing::info!(
            "Iris Store: Fetching a batch of {} Irises",
            identifiers.len()
        );

        // Conversion required for sql interpolation.
        let identifiers: Vec<i64> = identifiers.into_iter().map(|x| x as i64).collect();

        let irises = sqlx::query_as::<_, DbStoredIris>(
            r#"
            SELECT * FROM irises
            ORDER BY id ASC
            WHERE id = ANY($1)
            "#,
        )
        .bind(&identifiers)
        .fetch_all(&self.pool)
        .await?;

        Ok(irises)
    }

    /// Fetches first row from Iris table matched by id.
    ///
    /// # Arguments
    ///
    /// * `serial_id` - Serial ID of Iris to be fetched.
    ///
    /// # Returns
    ///
    /// Maybe a `DbStoredIris` instance.
    ///
    pub async fn fetch_iris_by_serial_id(
        &self,
        serial_id: u64,
    ) -> sqlx::Result<DbStoredIris, sqlx::Error> {
        tracing::info!(
            "PostgreSQL Store: Fetching Iris by serial-id ({})",
            serial_id
        );

        // Conversion required for sql interpolation.
        let id_of_iris = serial_id as i32;

        Ok(
            sqlx::query_as::<_, DbStoredIris>("SELECT * FROM irises WHERE id = $1")
                .bind(id_of_iris)
                .fetch_one(&self.pool)
                .await
                .expect("DB operation failure :: Fetch Iris by ID."),
        )
    }

    /// Stream irises in order.
    /// (only for testing) Stream irises in order.
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
    ) -> impl Stream<Item = Result<DbStoredIris>> + '_ {
        let count = self.count_irises().await.expect("Failed count_irises");
        let partition_size = count.div_ceil(partitions).max(1);

        let mut partition_streams = Vec::new();
        for i in 0..partitions {
            // we start from ID 1
            let start_id = 1 + partition_size * i;
            let end_id = start_id + partition_size - 1;

            // This base query yields `DbStoredIris`
            let stream = match min_last_modified_at {
                Some(min_last_modified_at) => sqlx::query_as::<_, DbStoredIris>(
                    "SELECT id, version_id, left_code, left_mask, right_code, right_mask FROM irises WHERE id \
                     BETWEEN $1 AND $2 AND last_modified_at >= $3",
                )
                .bind(start_id as i64)
                .bind(end_id as i64)
                .bind(min_last_modified_at)
                .fetch(&self.pool)
                .map_err(Into::into),
                None => sqlx::query_as::<_, DbStoredIris>(
                    "SELECT id, version_id, left_code, left_mask, right_code, right_mask FROM irises WHERE id \
                     BETWEEN $1 AND $2",
                )
                .bind(start_id as i64)
                .bind(end_id as i64)
                .fetch(&self.pool)
                .map_err(Into::into),
            }
            .boxed();

            partition_streams.push(stream);
        }

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
        external_tx: Option<&mut Transaction<'_, Postgres>>,
        id: i64,
        left_iris_share: &GaloisRingIrisCodeShare,
        left_mask_share: &GaloisRingTrimmedMaskCodeShare,
        right_iris_share: &GaloisRingIrisCodeShare,
        right_mask_share: &GaloisRingTrimmedMaskCodeShare,
    ) -> Result<()> {
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

        match external_tx {
            Some(external_tx) => {
                query.execute(external_tx.deref_mut()).await?;
            }
            None => {
                let mut new_tx = self.pool.begin().await?;
                query.execute(&mut *new_tx).await?;
                new_tx.commit().await?;
            }
        }

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
        let id: (Option<i64>,) = sqlx::query_as("SELECT MAX(id) FROM irises")
            .fetch_one(&self.pool)
            .await?;
        Ok(id.0.unwrap_or(0) as usize)
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

    pub async fn insert_modification(
        &self,
        serial_id: i64,
        request_type: &str,
        s3_url: Option<&str>,
    ) -> Result<Modification> {
        let persisted = false;
        let inserted: StoredModification = sqlx::query_as::<_, StoredModification>(
            r#"
            INSERT INTO modifications (serial_id, request_type, s3_url, status, persisted)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING
                id,
                serial_id,
                request_type,
                s3_url,
                status,
                persisted,
                result_message_body
            "#,
        )
        .bind(serial_id)
        .bind(request_type)
        .bind(s3_url)
        .bind(ModificationStatus::InProgress.to_string())
        .bind(persisted)
        .fetch_one(&self.pool)
        .await?;

        Ok(inserted.into())
    }

    pub async fn last_modifications(&self, count: usize) -> Result<Vec<Modification>> {
        let rows = sqlx::query_as::<_, StoredModification>(
            r#"
            SELECT
                id,
                serial_id,
                request_type,
                s3_url,
                status,
                persisted,
                result_message_body
            FROM modifications
            ORDER BY id DESC
            LIMIT $1
            "#,
        )
        .bind(count as i64)
        .fetch_all(&self.pool)
        .await?;

        let modifications = rows.into_iter().map(Into::into).collect();
        Ok(modifications)
    }

    /// Update the status, persisted flag, and result_message_body of the
    /// modifications based on their id.
    pub async fn update_modifications(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        modifications: &[&Modification],
    ) -> Result<(), sqlx::Error> {
        if modifications.is_empty() {
            return Ok(());
        }

        let ids: Vec<i64> = modifications.iter().map(|m| m.id).collect();
        let statuses: Vec<String> = modifications.iter().map(|m| m.status.clone()).collect();
        let persisteds: Vec<bool> = modifications.iter().map(|m| m.persisted).collect();
        let result_message_bodies: Vec<Option<String>> = modifications
            .iter()
            .map(|m| m.result_message_body.clone())
            .collect();

        sqlx::query(
            r#"
            UPDATE modifications
            SET status = data.status,
                persisted = data.persisted,
                result_message_body = data.result_message_body
            FROM (
                SELECT
                    unnest($1::bigint[])  as id,
                    unnest($2::text[])    as status,
                    unnest($3::bool[])    as persisted,
                    unnest($4::text[])    as result_message_body
            ) as data
            WHERE modifications.id = data.id
            "#,
        )
        .bind(&ids)
        .bind(&statuses)
        .bind(&persisteds)
        .bind(&result_message_bodies)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Delete modifications based on their id.
    pub async fn delete_modifications(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        modifications: &[Modification],
    ) -> Result<()> {
        if modifications.is_empty() {
            return Ok(());
        }

        // Extract the IDs from the modifications.
        let ids: Vec<i64> = modifications.iter().map(|m| m.id).collect();
        tracing::warn!(
            "Deleting modifications {:?} with IDs: {:?}",
            modifications,
            ids
        );

        // Execute a bulk delete using the ANY clause.
        sqlx::query(
            r#"
            DELETE FROM modifications
            WHERE id = ANY($1::bigint[])
            "#,
        )
        .bind(&ids)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
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
            self.insert_irises(
                &mut tx,
                &[StoredIrisRef {
                    id: (i + 1) as i64,
                    left_code: &share.coefs,
                    left_mask: &mask.coefs,
                    right_code: &share.coefs,
                    right_mask: &mask.coefs,
                }],
            )
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

fn cast_u8_to_u16(s: &[u8]) -> &[u16] {
    if s.is_empty() {
        &[] // A literal empty &[u8] may be unaligned.
    } else {
        cast_slice(s)
    }
}

#[cfg(test)]
#[cfg(feature = "db_dependent")]
pub mod tests {
    use super::{test_utils::*, *};
    use futures::TryStreamExt;
    use iris_mpc_common::{
        helpers::{
            smpc_request::{IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE},
            smpc_response::UniquenessResult,
            sync::ModificationStatus,
        },
        postgres::AccessMode,
    };

    const MAX_CONNECTIONS: u32 = 100;

    #[tokio::test]
    async fn test_store() -> Result<()> {
        // Create a unique schema for this test.
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let got: Vec<DbStoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let got: Vec<DbStoredIris> = store
            .stream_irises_par(Some(0), 2)
            .await
            .try_collect()
            .await?;
        assert_eq!(got.len(), 0);

        let codes_and_masks = &[
            StoredIrisRef {
                id: 1,
                left_code: &[1, 2, 3, 4],
                left_mask: &[5, 6, 7, 8],
                right_code: &[9, 10, 11, 12],
                right_mask: &[13, 14, 15, 16],
            },
            StoredIrisRef {
                id: 2,
                left_code: &[1117, 18, 19, 20],
                left_mask: &[21, 1122, 23, 24],
                right_code: &[25, 26, 1127, 28],
                right_mask: &[29, 30, 31, 1132],
            },
            StoredIrisRef {
                id: 3,
                left_code: &[17, 18, 19, 20],
                left_mask: &[21, 22, 23, 24],
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
            .try_collect()
            .await?;
        got_par.sort_by_key(|iris| iris.id);
        assert_eq!(got, got_par);

        assert_eq!(got_len, 3);
        assert_eq!(got_par.len(), 3);
        assert_eq!(got.len(), 3);

        for i in 0..3 {
            assert_eq!(got[i].serial_id(), i + 1);
            assert_eq!(got[i].version_id(), 0);
            assert_eq!(got[i].vector_id(), VectorId::new(i as u32 + 1, 0));
            assert_eq!(got[i].left_code(), codes_and_masks[i].left_code);
            assert_eq!(got[i].left_mask(), codes_and_masks[i].left_mask);
            assert_eq!(got[i].right_code(), codes_and_masks[i].right_code);
            assert_eq!(got[i].right_mask(), codes_and_masks[i].right_mask);
        }

        // Clean up on success.
        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_insert() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &[]).await?;
        store.insert_irises(&mut tx, &[]).await?;
        tx.commit().await?;
        store.mark_requests_deleted(&[]).await?;

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_many() -> Result<()> {
        let count: usize = 1 << 3;

        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let mut codes_and_masks = vec![];

        for i in 0..count {
            let iris = StoredIrisRef {
                id: (i + 1) as i64,
                left_code: &[123_u16; 12800],
                left_mask: &[456_u16; 12800],
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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
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
                .try_collect()
                .await?;
            got_par.sort_by_key(|iris| iris.id);
            assert_eq!(got, got_par);
        }

        let got = store.last_results(count).await?;
        assert_eq!(got, result_events);

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_init_db_with_random_shares() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let expected_generated_irises_num = 10;
        store
            .init_db_with_random_shares(0, 0, expected_generated_irises_num, true)
            .await?;

        let generated_irises_count = store.count_irises().await?;
        assert_eq!(generated_irises_count, expected_generated_irises_num);

        cleanup(&postgres_client, &schema_name).await
    }

    #[tokio::test]
    async fn test_rollback() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let mut irises = vec![];
        for i in 0..10 {
            let iris = StoredIrisRef {
                id: (i + 1) as i64,
                left_code: &[123_u16; 12800],
                left_mask: &[456_u16; 12800],
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

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_results() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        let result_events = vec!["event1".to_string(), "event2".to_string()];
        let mut tx = store.tx().await?;
        store.insert_results(&mut tx, &result_events).await?;
        store.insert_results(&mut tx, &result_events).await?;
        tx.commit().await?;

        let got = store.last_results(2).await?;
        assert_eq!(got, vec!["event1", "event2"]);

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_mark_requests_deleted() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        assert_eq!(store.last_deleted_requests(2).await?.len(), 0);

        for i in 0..2 {
            let request_ids = (0..2)
                .map(|j| format!("test_{}_{}", i, j))
                .collect::<Vec<_>>();
            store.mark_requests_deleted(&request_ids).await?;

            let got = store.last_deleted_requests(2).await?;
            assert_eq!(got, request_ids);
        }

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_update_iris() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // insert two irises into db
        let iris1 = StoredIrisRef {
            id: 1,
            left_code: &[123_u16; 12800],
            left_mask: &[456_u16; 6400],
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
            id: 1,
            coefs: [666_u16; 12800],
        };
        let updated_left_mask = GaloisRingTrimmedMaskCodeShare {
            id: 1,
            coefs: [777_u16; 6400],
        };
        let updated_right_code = GaloisRingIrisCodeShare {
            id: 1,
            coefs: [888_u16; 12800],
        };
        let updated_right_mask = GaloisRingTrimmedMaskCodeShare {
            id: 1,
            coefs: [999_u16; 6400],
        };
        store
            .update_iris(
                None,
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
        assert_eq!(got[0].version_id(), 1);

        // assert the other iris in db is not updated
        assert_eq!(cast_u8_to_u16(&got[1].left_code), iris2.left_code);
        assert_eq!(cast_u8_to_u16(&got[1].left_mask), iris2.left_mask);
        assert_eq!(cast_u8_to_u16(&got[1].right_code), iris2.right_code);
        assert_eq!(cast_u8_to_u16(&got[1].right_mask), iris2.right_mask);
        assert_eq!(got[1].version_id(), 0);

        // update with the same values and expect the version not to change
        store
            .update_iris(
                None,
                1,
                &updated_left_code,
                &updated_left_mask,
                &updated_right_code,
                &updated_right_mask,
            )
            .await?;

        let got_second_update: Vec<DbStoredIris> =
            store.stream_irises().await.try_collect().await?;
        assert_eq!(got_second_update.len(), 2);
        assert_eq!(
            cast_u8_to_u16(&got_second_update[0].left_code),
            updated_left_code.coefs
        );
        assert_eq!(
            cast_u8_to_u16(&got_second_update[0].left_mask),
            updated_left_mask.coefs
        );
        assert_eq!(
            cast_u8_to_u16(&got_second_update[0].right_code),
            updated_right_code.coefs
        );
        assert_eq!(
            cast_u8_to_u16(&got_second_update[0].right_mask),
            updated_right_mask.coefs
        );
        assert_eq!(got_second_update[0].version_id(), 1);

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_modification() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // 1. Insert a new modification
        let inserted = store
            .insert_modification(42, IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // 2. Check that we got a valid result
        assert_modification(
            &inserted,
            1,
            42,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );

        // 3. Insert another modification
        let inserted = store
            .insert_modification(43, REAUTH_MESSAGE_TYPE, Some("https://example.com"))
            .await?;

        // 4. Check that we got a valid result
        assert_modification(
            &inserted,
            2,
            43,
            REAUTH_MESSAGE_TYPE,
            Some("https://example.com".to_string()),
            ModificationStatus::InProgress,
            false,
            None,
        );

        // 5. Clean up
        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_last_modifications() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert a few modifications
        for serial_id in 11..=15 {
            store
                .insert_modification(serial_id, IDENTITY_DELETION_MESSAGE_TYPE, None)
                .await?;
        }

        // Retrieve the last 2 modifications
        let last_two = store.last_modifications(2).await?;
        assert_eq!(last_two.len(), 2);
        let last = &last_two[0];
        let second_last = &last_two[1];

        // Assert results
        assert_modification(
            last,
            5,
            15,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );
        assert_modification(
            second_last,
            4,
            14,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
        );

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_modification(
        actual: &Modification,
        expected_id: i64,
        expected_serial_id: i64,
        expected_request_type: &str,
        expected_s3_url: Option<String>,
        expected_status: ModificationStatus,
        expected_persisted: bool,
        expected_result_body: Option<String>,
    ) {
        assert_eq!(actual.id, expected_id);
        assert_eq!(actual.serial_id, expected_serial_id);
        assert_eq!(actual.request_type, expected_request_type);
        assert_eq!(actual.s3_url, expected_s3_url);
        assert_eq!(actual.status, expected_status.to_string());
        assert_eq!(actual.persisted, expected_persisted);
        assert_eq!(actual.result_message_body, expected_result_body);
    }

    #[tokio::test]
    async fn test_update_modifications() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert three modifications
        let mut m1 = store
            .insert_modification(100, IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;
        let mut m2 = store
            .insert_modification(50, REAUTH_MESSAGE_TYPE, Some("http://example.com/50"))
            .await?;
        let _m3 = store
            .insert_modification(150, REAUTH_MESSAGE_TYPE, Some("http://example.com/150"))
            .await?;

        // Update the status & persisted fields for first two in a single transaction
        let mut tx = store.tx().await?;
        m1.mark_completed(true, "m1");
        m2.mark_completed(false, "m2");

        let modifications_to_update = vec![&m1, &m2];
        store
            .update_modifications(&mut tx, &modifications_to_update)
            .await?;

        tx.commit().await?;

        // Check that the DB is updated
        let last_three = store.last_modifications(3).await?;
        assert_eq!(last_three.len(), 3);
        assert_modification(
            &last_three[0],
            3,
            150,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/150".to_string()),
            ModificationStatus::InProgress,
            false,
            None,
        );
        assert_modification(
            &last_three[1],
            2,
            50,
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/50".to_string()),
            ModificationStatus::Completed,
            false,
            Some("m2".to_string()),
        );
        assert_modification(
            &last_three[2],
            1,
            100,
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some("m1".to_string()),
        );

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_modifications() -> Result<()> {
        // Set up a temporary schema and a new store.
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert three modifications.
        let mut m1 = store
            .insert_modification(11, IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;
        let m2 = store
            .insert_modification(12, REAUTH_MESSAGE_TYPE, Some("http://example.com/12"))
            .await?;
        let m3 = store
            .insert_modification(13, IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // mark m1 as completed
        m1.mark_completed(true, "m1");
        let mut tx = store.tx().await?;
        store.update_modifications(&mut tx, &[&m1]).await?;
        tx.commit().await?;

        // Verify that all three modifications exist.
        let all_mods = store.last_modifications(5).await?;
        assert_eq!(all_mods.len(), 3);

        // Begin a transaction and delete modifications m2 and m3.
        let mut tx = store.tx().await?;
        store
            .delete_modifications(&mut tx, &[m2.clone(), m3.clone()])
            .await?;
        tx.commit().await?;

        // Fetch the remaining modifications.
        let remaining = store.last_modifications(5).await?;
        // We expect only one modification to remain (m1).
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, m1.id);

        // Clean up the temporary schema.
        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    fn assert_contiguous_id(vec: &[DbStoredIris]) {
        assert!(
            vec.iter()
                .enumerate()
                .all(|(i, row)| row.id == (i + 1) as i64),
            "IDs must be contiguous and in order"
        );
    }
}

pub mod test_utils {
    use super::*;
    const APP_NAME: &str = "SMPC";
    const DOTENV_TEST: &str = ".env.test";

    pub fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(Config::load_config(APP_NAME)?
            .database
            .ok_or(eyre!("Missing database config"))?
            .url)
    }

    pub fn temporary_name() -> String {
        format!("SMPC_test{}_0", rand::random::<u32>())
    }

    pub async fn cleanup(postgres_client: &PostgresClient, schema_name: &str) -> Result<()> {
        assert!(schema_name.starts_with("SMPC_test"));
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
            .execute(&postgres_client.pool)
            .await?;
        Ok(())
    }
}
