pub mod loader;
pub mod rerand;
mod s3_importer;

use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::{
    stream::{self},
    Stream, StreamExt, TryStreamExt,
};
use iris_mpc_common::helpers::sync::MOD_STATUS_IN_PROGRESS;
use iris_mpc_common::{
    config::Config,
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        smpc_request::{
            IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_UPDATE_MESSAGE_TYPE,
        },
        sync::Modification,
    },
    iris_db::iris::IrisCode,
    postgres::PostgresClient,
    vector_id::{SerialId, VectorId},
};
use itertools::izip;
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
        VectorId::new(self.id as u32, self.version_id)
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
    /// Not really intended to be used directly, use StoredIrisRef instead.
    pub fn new(
        id: i64,
        version_id: i16,
        left_code: Vec<u8>,
        left_mask: Vec<u8>,
        right_code: Vec<u8>,
        right_mask: Vec<u8>,
    ) -> Self {
        Self {
            id,
            version_id,
            left_code,
            left_mask,
            right_code,
            right_mask,
        }
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

// Convertor: DbStoredIris -> IrisIdentifiers.
impl From<&DbStoredIris> for VectorId {
    fn from(value: &DbStoredIris) -> Self {
        VectorId::new(value.serial_id() as SerialId, value.version_id())
    }
}

#[derive(sqlx::FromRow, Debug, Default)]
pub struct StoredModification {
    pub id: i64,
    pub serial_id: Option<i64>,
    pub request_type: String,
    pub s3_url: Option<String>,
    pub status: String,
    pub persisted: bool,
    pub result_message_body: Option<String>,
    pub graph_mutation: Option<Vec<u8>>,
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
            graph_mutation: stored.graph_mutation,
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

    pub async fn get_iris_data_by_id(&self, id: i64) -> Result<DbStoredIris> {
        let iris = sqlx::query_as::<_, DbStoredIris>(
            r#"
            SELECT *
            FROM irises
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_one(&self.pool)
        .await?;
        Ok(iris)
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

    pub async fn insert_copy_irises(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        vector_ids: &[VectorId],
        codes_and_masks: &[StoredIrisRef<'_>],
    ) -> Result<Vec<i64>> {
        if codes_and_masks.is_empty() {
            return Ok(vec![]);
        }
        if vector_ids.len() != codes_and_masks.len() {
            return Err(eyre!(
                "vector_ids and codes_and_masks must have the same length"
            ));
        }
        let mut query = sqlx::QueryBuilder::new(
            "INSERT INTO irises (id, version_id, left_code, left_mask, right_code, right_mask)",
        );
        query.push_values(
            codes_and_masks.iter().zip(vector_ids.iter()),
            |mut query, (iris, vector_id)| {
                query.push_bind(iris.id);
                query.push_bind(vector_id.version_id());
                query.push_bind(cast_slice::<u16, u8>(iris.left_code));
                query.push_bind(cast_slice::<u16, u8>(iris.left_mask));
                query.push_bind(cast_slice::<u16, u8>(iris.right_code));
                query.push_bind(cast_slice::<u16, u8>(iris.right_mask));
            },
        );

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

    // Update existing iris with given shares.
    pub async fn update_iris_with_version_id(
        &self,
        external_tx: Option<&mut Transaction<'_, Postgres>>,
        version_id: i16,
        codes_and_masks: &StoredIrisRef<'_>,
    ) -> Result<()> {
        let query = sqlx::query(
            r#"
UPDATE irises SET (version_id, left_code, left_mask, right_code, right_mask) = ($2, $3, $4, $5  , $6)
WHERE id = $1;
"#,
        )
        .bind(codes_and_masks.id)
        .bind(version_id)
        .bind(cast_slice::<u16, u8>(codes_and_masks.left_code))
        .bind(cast_slice::<u16, u8>(codes_and_masks.left_mask))
        .bind(cast_slice::<u16, u8>(codes_and_masks.right_code))
        .bind(cast_slice::<u16, u8>(codes_and_masks.right_mask));

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

    pub async fn insert_modification(
        &self,
        serial_id: Option<i64>,
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
                result_message_body,
                graph_mutation
            "#,
        )
        .bind(serial_id)
        .bind(request_type)
        .bind(s3_url)
        .bind(MOD_STATUS_IN_PROGRESS)
        .bind(persisted)
        .fetch_one(&self.pool)
        .await?;

        tracing::info!(
            "Inserted {} modification: id={:?}, serial_id={:?}, request_type={}",
            MOD_STATUS_IN_PROGRESS,
            inserted.id,
            serial_id,
            request_type
        );

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
                result_message_body,
                graph_mutation
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

    /// Fetch modifications updated after a certain ID that are less than a serial id.
    /// This is for the genesis protocol to fetch modifications that need to be indexed.
    ///
    /// # Arguments
    ///
    /// * `after_modification_id` - Modification identifier from which to filter result.
    /// * `serial_id_less_than` - Iris serial identifier to which to filter result.
    ///
    /// # Returns
    ///
    /// An ordered vector of `Modification` instances in ascending order of their IDs.
    /// Serial ids are initialized for all returned modifications.
    ///
    pub async fn get_persisted_modifications_after_id(
        &self,
        after_modification_id: i64,
        serial_id_less_than: u32,
    ) -> Result<(Vec<Modification>, Option<i64>)> {
        let message_types = &[
            RESET_UPDATE_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE,
            IDENTITY_DELETION_MESSAGE_TYPE,
        ];
        let mut tx = self.pool.begin().await?;

        let rows = sqlx::query_as::<_, StoredModification>(
            r#"
            SELECT
                id,
                serial_id,
                request_type,
                s3_url,
                status,
                persisted,
                result_message_body,
                graph_mutation
            FROM modifications
            WHERE id > $1
              AND request_type = ANY($2)
              AND persisted = true
              AND status = 'COMPLETED'
              AND serial_id <= $3
            ORDER BY id ASC
            "#,
        )
        .bind(after_modification_id)
        .bind(message_types)
        .bind(serial_id_less_than as i64)
        .fetch_all(&mut *tx)
        .await?;

        // Fetch the max id from modifications that are persisted and completed.
        let max_id: Option<i64> = sqlx::query_scalar(
            r#"
            SELECT MAX(id) FROM modifications
            WHERE persisted = true
                AND request_type = ANY($2)
                AND status = 'COMPLETED'
            "#,
        )
        .bind(after_modification_id)
        .bind(message_types)
        .fetch_one(&mut *tx)
        .await?;

        let modifications = rows.into_iter().map(Into::into).collect();
        Ok((modifications, max_id))
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
        let persisted: Vec<bool> = modifications.iter().map(|m| m.persisted).collect();
        let result_message_bodies: Vec<Option<String>> = modifications
            .iter()
            .map(|m| m.result_message_body.clone())
            .collect();
        let serial_ids: Vec<Option<i64>> = modifications.iter().map(|m| m.serial_id).collect();
        let graph_mutations: Vec<Option<Vec<u8>>> = modifications
            .iter()
            .map(|m| m.graph_mutation.clone())
            .collect();

        for (id, status, persisted, serial_id) in izip!(&ids, &statuses, &persisted, &serial_ids) {
            tracing::info!(
                "Updating modification id={} with status={}, persisted={}, serial_id={:?}",
                id,
                status,
                persisted,
                serial_id
            );
        }

        sqlx::query(
            r#"
            UPDATE modifications
            SET status = data.status,
                persisted = data.persisted,
                result_message_body = data.result_message_body,
                serial_id = data.serial_id,
                graph_mutation = data.graph_mutation
            FROM (
                SELECT
                    unnest($1::bigint[])  as id,
                    unnest($2::text[])    as status,
                    unnest($3::bool[])    as persisted,
                    unnest($4::text[])    as result_message_body,
                    unnest($5::bigint[])  as serial_id,
                    unnest($6::bytea[])   as graph_mutation
            ) as data
            WHERE modifications.id = data.id
            "#,
        )
        .bind(&ids)
        .bind(&statuses)
        .bind(&persisted)
        .bind(&result_message_bodies)
        .bind(&serial_ids)
        .bind(&graph_mutations)
        .execute(tx.deref_mut())
        .await?;

        Ok(())
    }

    /// Insert a modification recovered from a peer. Uses the peer's `id` to
    /// keep modification IDs consistent across parties. If the `id` already
    /// exists, updates the row to match the peer's state.
    pub async fn upsert_recovered_modification(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        m: &Modification,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO modifications (id, serial_id, request_type, s3_url, status, persisted, result_message_body, graph_mutation)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                persisted = EXCLUDED.persisted,
                result_message_body = EXCLUDED.result_message_body,
                serial_id = EXCLUDED.serial_id,
                graph_mutation = EXCLUDED.graph_mutation
            "#,
        )
        .bind(m.id)
        .bind(m.serial_id)
        .bind(&m.request_type)
        .bind(&m.s3_url)
        .bind(&m.status)
        .bind(m.persisted)
        .bind(&m.result_message_body)
        .bind(&m.graph_mutation)
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

    /// Delete all modifications from the modifications table.
    pub async fn clear_modifications_table(
        &self,
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<()> {
        sqlx::query("DELETE FROM modifications")
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
            smpc_request::{
                IDENTITY_DELETION_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
            },
            sync::ModificationStatus,
        },
        postgres::AccessMode,
    };

    // Max connections default to 100 for Postgres, but can't test at quite this level when running
    // multiple DB-related tests in parallel.
    const MAX_CONNECTIONS: u32 = 80;

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
        store.insert_irises(&mut tx, &[]).await?;
        tx.commit().await?;

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

        let mut tx = store.tx().await?;
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

        // Test get_iris_data for id 1
        let fetched_iris1 = store.get_iris_data_by_id(1).await?;
        assert_eq!(fetched_iris1.id, 1);
        assert_eq!(fetched_iris1.left_code(), &[123_u16; 12800]);
        assert_eq!(fetched_iris1.left_mask(), &[456_u16; 6400]);
        assert_eq!(fetched_iris1.right_code(), &[789_u16; 12800]);
        assert_eq!(fetched_iris1.right_mask(), &[101_u16; 6400]);

        // Test get_iris_data for id 2
        let fetched_iris2 = store.get_iris_data_by_id(2).await?;
        assert_eq!(fetched_iris2.id, 2);
        assert_eq!(fetched_iris2.left_code(), &[123_u16; 12800]);
        assert_eq!(fetched_iris2.left_mask(), &[456_u16; 6400]);
        assert_eq!(fetched_iris2.right_code(), &[789_u16; 12800]);
        assert_eq!(fetched_iris2.right_mask(), &[101_u16; 6400]);

        // Test get_iris_data for non-existent id (should error)
        let not_found = store.get_iris_data_by_id(999).await;
        assert!(not_found.is_err());

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
            .insert_modification(Some(42), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // 2. Check that we got a valid result
        assert_modification(
            &inserted,
            1,
            Some(42),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
            None,
        );

        // 3. Insert another modification
        let inserted = store
            .insert_modification(Some(43), REAUTH_MESSAGE_TYPE, Some("https://example.com"))
            .await?;

        // 4. Check that we got a valid result
        assert_modification(
            &inserted,
            2,
            Some(43),
            REAUTH_MESSAGE_TYPE,
            Some("https://example.com".to_string()),
            ModificationStatus::InProgress,
            false,
            None,
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
                .insert_modification(Some(serial_id), IDENTITY_DELETION_MESSAGE_TYPE, None)
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
            Some(15),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
            None,
        );
        assert_modification(
            second_last,
            4,
            Some(14),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::InProgress,
            false,
            None,
            None,
        );

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_modification(
        actual: &Modification,
        expected_id: i64,
        expected_serial_id: Option<i64>,
        expected_request_type: &str,
        expected_s3_url: Option<String>,
        expected_status: ModificationStatus,
        expected_persisted: bool,
        expected_result_body: Option<String>,
        expected_graph_mut: Option<Vec<u8>>,
    ) {
        assert_eq!(actual.id, expected_id);
        assert_eq!(actual.serial_id, expected_serial_id);
        assert_eq!(actual.request_type, expected_request_type);
        assert_eq!(actual.s3_url, expected_s3_url);
        assert_eq!(actual.status, expected_status.to_string());
        assert_eq!(actual.persisted, expected_persisted);
        assert_eq!(actual.result_message_body, expected_result_body);
        assert_eq!(actual.graph_mutation, expected_graph_mut);
    }

    #[tokio::test]
    async fn test_update_modifications() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert five modifications
        let mut m1 = store
            .insert_modification(Some(100), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;
        let mut m2 = store
            .insert_modification(Some(50), REAUTH_MESSAGE_TYPE, Some("http://example.com/50"))
            .await?;
        let mut m3 = store
            .insert_modification(None, UNIQUENESS_MESSAGE_TYPE, Some("http://example.com/m3"))
            .await?;
        let mut m4 = store
            .insert_modification(None, UNIQUENESS_MESSAGE_TYPE, Some("http://example.com/m4"))
            .await?;
        let _m5 = store
            .insert_modification(
                Some(150),
                REAUTH_MESSAGE_TYPE,
                Some("http://example.com/150"),
            )
            .await?;
        let m1_graph_mut = vec![1u8, 2u8, 3u8, 4u8];
        let m3_graph_mut = vec![3u8, 123u8, 34u8, 99u8];
        // Update the status & persisted fields for first four in a single transaction
        let mut tx = store.tx().await?;
        m1.mark_completed(true, "m1", None, Some(m1_graph_mut.clone()));
        m2.mark_completed(false, "m2", None, None);
        m3.mark_completed(true, "m3", Some(101), Some(m3_graph_mut.clone()));
        m4.mark_completed(false, "m4", None, None);

        let modifications_to_update = vec![&m1, &m2, &m3, &m4];
        store
            .update_modifications(&mut tx, &modifications_to_update)
            .await?;

        tx.commit().await?;

        // Check that the DB is updated
        let last_five = store.last_modifications(5).await?;
        assert_eq!(last_five.len(), 5);
        assert_modification(
            &last_five[0],
            5,
            Some(150),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/150".to_string()),
            ModificationStatus::InProgress,
            false,
            None,
            None,
        );
        assert_modification(
            &last_five[1],
            4,
            None,
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/m4".to_string()),
            ModificationStatus::Completed,
            false,
            Some("m4".to_string()),
            None,
        );
        assert_modification(
            &last_five[2],
            3,
            Some(101),
            UNIQUENESS_MESSAGE_TYPE,
            Some("http://example.com/m3".to_string()),
            ModificationStatus::Completed,
            true,
            Some("m3".to_string()),
            Some(m3_graph_mut.clone()),
        );
        assert_modification(
            &last_five[3],
            2,
            Some(50),
            REAUTH_MESSAGE_TYPE,
            Some("http://example.com/50".to_string()),
            ModificationStatus::Completed,
            false,
            Some("m2".to_string()),
            None,
        );
        assert_modification(
            &last_five[4],
            1,
            Some(100),
            IDENTITY_DELETION_MESSAGE_TYPE,
            None,
            ModificationStatus::Completed,
            true,
            Some("m1".to_string()),
            Some(m1_graph_mut.clone()),
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
            .insert_modification(Some(11), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;
        let m2 = store
            .insert_modification(Some(12), REAUTH_MESSAGE_TYPE, Some("http://example.com/12"))
            .await?;
        let m3 = store
            .insert_modification(Some(13), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // mark m1 as completed
        m1.mark_completed(true, "m1", None, None);
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

    #[tokio::test]
    async fn test_clear_modifications_table() -> Result<()> {
        // Set up a temporary schema and a new store.
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert several modifications.
        for i in 0..5 {
            store
                .insert_modification(Some(100 + i), IDENTITY_DELETION_MESSAGE_TYPE, None)
                .await?;
        }

        // Ensure modifications are present.
        let all_mods = store.last_modifications(10).await?;
        assert_eq!(all_mods.len(), 5);

        // Clear the modifications table.
        let mut tx = store.tx().await?;
        store.clear_modifications_table(&mut tx).await?;
        tx.commit().await?;

        // Ensure the table is empty.
        let mods_after_clear = store.last_modifications(10).await?;
        assert_eq!(mods_after_clear.len(), 0);

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

    #[tokio::test]
    async fn test_fetch_update_modifications_from() -> Result<()> {
        let schema_name = temporary_name();
        let postgres_client =
            PostgresClient::new(test_db_url()?.as_str(), &schema_name, AccessMode::ReadWrite)
                .await?;
        let store = Store::new(&postgres_client).await?;

        // Insert a variety of modifications with different request types
        // ID 1: identity_deletion (should be included when persisted = true)
        let mod1 = store
            .insert_modification(Some(100), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // ID 2: reauth (should be included when persisted = true)
        let mod2 = store
            .insert_modification(
                Some(101),
                REAUTH_MESSAGE_TYPE,
                Some("http://example.com/reauth"),
            )
            .await?;

        // ID 3: reset_update (should not be included when persisted = false)
        store
            .insert_modification(
                Some(102),
                RESET_UPDATE_MESSAGE_TYPE,
                Some("http://example.com/reset"),
            )
            .await?;

        // ID 4: reset_update (should be included when persisted = true)
        let mod4 = store
            .insert_modification(
                Some(103),
                RESET_UPDATE_MESSAGE_TYPE,
                Some("http://example.com/reset"),
            )
            .await?;

        // ID 5: uniqueness (should NOT be included due to request type)
        store
            .insert_modification(
                Some(104),
                UNIQUENESS_MESSAGE_TYPE,
                Some("http://example.com/uniqueness"),
            )
            .await?;

        // ID 6: reset_check (should NOT be included due to request type)
        store
            .insert_modification(
                Some(105),
                RESET_CHECK_MESSAGE_TYPE,
                Some("http://example.com/reset_check"),
            )
            .await?;

        // ID 7: identity_deletion (should NOT be included due to persisted = false)
        let mod6 = store
            .insert_modification(Some(106), IDENTITY_DELETION_MESSAGE_TYPE, None)
            .await?;

        // Make modifications 1, 2, 4 persisted = true
        let mut mod1 = mod1;
        let mut mod2 = mod2;
        let mut mod4 = mod4;
        mod1.mark_completed(true, "result1", None, None);
        mod2.mark_completed(true, "result2", None, None);
        mod4.mark_completed(true, "result4", None, None);

        let mut tx = store.tx().await?;
        store
            .update_modifications(&mut tx, &[&mod1, &mod2, &mod4])
            .await?;
        tx.commit().await?;

        // Test 1: Get all modifications with ID > 0 (should return only the 3 persisted ones with valid request types)
        let (modifications, max_id) = store.get_persisted_modifications_after_id(0, 106).await?;
        assert_eq!(
            modifications.len(),
            3,
            "Should return 3 persisted modifications"
        );
        assert_eq!(max_id, Some(4), "Max ID should be 4");

        // Check the specific request types are correct
        let request_types: Vec<&str> = modifications
            .iter()
            .map(|m| m.request_type.as_str())
            .collect();
        assert!(
            request_types[0] == IDENTITY_DELETION_MESSAGE_TYPE,
            "Should include persisted identity_deletion requests"
        );
        assert!(
            request_types[1] == REAUTH_MESSAGE_TYPE,
            "Should include persisted reauth requests"
        );
        assert!(
            request_types[2] == RESET_UPDATE_MESSAGE_TYPE,
            "Should include persisted reset_update requests"
        );

        // Test 2: Get persisted modifications with ID > 2 (should exclude the first two)
        let (modifications, max_id) = store.get_persisted_modifications_after_id(2, 106).await?;
        assert_eq!(
            modifications.len(),
            1,
            "Should return 1 persisted modification"
        );
        assert_eq!(max_id, Some(4), "Max ID should be 4");

        // The IDs should be greater than 2
        for modification in &modifications {
            assert!(
                modification.id > 2,
                "Modification ID should be greater than 2"
            );
            assert!(modification.persisted, "Modification should be persisted");
        }

        // Make mod6 persisted=true
        let mut mod6 = mod6;
        mod6.mark_completed(true, "result6", None, None);

        let mut tx = store.tx().await?;
        store.update_modifications(&mut tx, &[&mod6]).await?;
        tx.commit().await?;

        // Test 3: Get persisted modifications with ID > 3 (should include ID 4 now)
        // Should not include the last serial id
        let (modifications, max_id) = store.get_persisted_modifications_after_id(3, 105).await?;
        assert_eq!(
            modifications.len(),
            1,
            "Should return 1 persisted modification"
        );
        assert_eq!(max_id, Some(7), "Max ID should be 7");
        assert_eq!(
            modifications[0].id, 4,
            "Should return modification with ID 4"
        );

        // Test 4: Get modifications with ID > 6 (should return none)
        let (modifications, max_id) = store.get_persisted_modifications_after_id(7, 106).await?;
        assert_eq!(modifications.len(), 0, "Should return 0 modifications");
        assert_eq!(max_id, Some(7), "Max ID should be 7");

        // Test 5: Get modifications with ID > 0 and serial id is less than 102
        let (modifications, max_id) = store.get_persisted_modifications_after_id(0, 102).await?;
        assert_eq!(
            modifications.len(),
            2,
            "Should return 2 persisted modifications"
        );
        assert_eq!(max_id, Some(7), "Max ID should be 7");
        assert_eq!(
            modifications[0].id, 1,
            "Should return modification with ID 1"
        );
        assert_eq!(
            modifications[1].id, 2,
            "Should return modification with ID 2"
        );

        cleanup(&postgres_client, &schema_name).await?;
        Ok(())
    }
}

pub mod test_utils {
    use super::*;
    const SCHEMA_NAME: &str = "SMPC";
    const DOTENV_TEST: &str = ".env.test";

    pub fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(Config::load_config(SCHEMA_NAME)?
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
