//! Minimal database primitives for continuous rerandomization.

use std::collections::{HashMap, HashSet};

use bytemuck::cast_slice;
use eyre::{ensure, eyre, Result};
use iris_mpc_common::rerand_offsets::{retarget_shares, EpochKey, EpochSeed};
use sqlx::{pool::PoolConnection, PgPool, Postgres};

use crate::{DbStoredIris, S3StoredIris};

pub struct RerandRowUpdate {
    pub id: i64,
    pub expected_version_id: i16,
    pub expected_semantic_id: Option<String>,
    pub from_epoch: i32,
    pub left_code: Vec<u16>,
    pub left_mask: Vec<u16>,
    pub right_code: Vec<u16>,
    pub right_mask: Vec<u16>,
}

#[derive(sqlx::FromRow)]
pub struct RerandSourceRow {
    #[sqlx(flatten)]
    pub iris: DbStoredIris,
    pub semantic_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerandStoreState {
    pub store_id: String,
    pub environment: String,
    pub coordination_id: String,
    pub party_id: u8,
    pub store_kind: String,
    pub writer_role: String,
    pub last_completed_epoch: u32,
    pub last_seed_commitment: Option<[u8; 32]>,
    pub active_epoch: Option<u32>,
    pub active_seed_commitment: Option<[u8; 32]>,
    pub next_id: Option<i64>,
    pub max_id: Option<i64>,
}

/// Verify the immutable identity before a loader or worker uses this database.
pub async fn verify_store_identity(
    pool: &PgPool,
    expected_store_id: &str,
    expected_environment: &str,
    expected_coordination_id: &str,
    expected_party_id: u8,
    expected_store_kind: &str,
) -> Result<RerandStoreState> {
    ensure!(
        !expected_store_id.is_empty()
            && !expected_environment.is_empty()
            && !expected_coordination_id.is_empty(),
        "rerandomization store identity is empty"
    );
    let row = sqlx::query_as::<
        _,
        (
            Option<String>,
            Option<String>,
            Option<String>,
            Option<i16>,
            Option<String>,
            Option<String>,
            i32,
            Option<Vec<u8>>,
            Option<i32>,
            Option<Vec<u8>>,
            Option<i64>,
            Option<i64>,
        ),
    >("SELECT * FROM get_rerand_store_state()")
    .fetch_one(pool)
    .await?;
    let store_id = row
        .0
        .ok_or_else(|| eyre!("rerandomization store identity has not been initialized"))?;
    let environment = row
        .1
        .ok_or_else(|| eyre!("rerandomization environment is uninitialized"))?;
    let coordination_id = row
        .2
        .ok_or_else(|| eyre!("rerandomization coordination ID is uninitialized"))?;
    let party_id: u8 = row
        .3
        .ok_or_else(|| eyre!("rerandomization party is uninitialized"))?
        .try_into()?;
    let store_kind = row
        .4
        .ok_or_else(|| eyre!("rerandomization store kind is uninitialized"))?;
    ensure!(
        store_id == expected_store_id
            && environment == expected_environment
            && coordination_id == expected_coordination_id
            && party_id == expected_party_id
            && store_kind == expected_store_kind,
        "rerandomization physical store identity mismatch"
    );
    Ok(RerandStoreState {
        store_id,
        environment,
        coordination_id,
        party_id,
        store_kind,
        writer_role: row
            .5
            .ok_or_else(|| eyre!("rerandomization writer is uninitialized"))?,
        last_completed_epoch: row.6.try_into()?,
        last_seed_commitment: parse_commitment(row.7)?,
        active_epoch: row.8.map(u32::try_from).transpose()?,
        active_seed_commitment: parse_commitment(row.9)?,
        next_id: row.10,
        max_id: row.11,
    })
}

pub async fn get_rerand_epoch_inventory(pool: &PgPool) -> Result<Vec<(i32, i64)>> {
    Ok(sqlx::query_as(
        "SELECT rerand_epoch, COUNT(*) FROM irises GROUP BY rerand_epoch ORDER BY rerand_epoch",
    )
    .fetch_all(pool)
    .await?)
}

/// A session-level singleton lock. All pass reads, state transitions, and CAS
/// writes use this exact backend because the database trigger verifies it.
pub struct RerandPass {
    connection: PoolConnection<Postgres>,
    pub state: RerandStoreState,
}

impl RerandPass {
    pub async fn acquire(
        pool: &PgPool,
        expected_store_id: &str,
        expected_environment: &str,
        expected_coordination_id: &str,
        expected_party_id: u8,
        expected_store_kind: &str,
    ) -> Result<Self> {
        let expected = verify_store_identity(
            pool,
            expected_store_id,
            expected_environment,
            expected_coordination_id,
            expected_party_id,
            expected_store_kind,
        )
        .await?;
        let mut connection = pool.acquire().await?;
        connection.close_on_drop();
        let acquired: bool = sqlx::query_scalar("SELECT try_rerand_pass_lock()")
            .fetch_one(&mut *connection)
            .await?;
        ensure!(acquired, "another rerandomization pass is already running");
        let state = load_state(&mut connection).await?;
        ensure!(
            same_identity(&state, &expected),
            "store identity changed while acquiring the pass lock"
        );
        Ok(Self { connection, state })
    }

    /// Start `last_completed + 1`, or resume that exact pass after a crash.
    pub async fn begin_or_resume(
        &mut self,
        epoch: u32,
        seed_commitment: [u8; 32],
    ) -> Result<(i64, i64)> {
        ensure!(epoch > 0 && epoch <= i32::MAX as u32, "invalid pass epoch");
        let epoch = epoch as i32;
        if let Some(active) = self.state.active_epoch {
            ensure!(active == epoch as u32, "a different pass is already active");
            ensure!(
                self.state.active_seed_commitment == Some(seed_commitment),
                "active pass seed commitment changed"
            );
        } else {
            ensure!(
                Some(epoch as u32) == self.state.last_completed_epoch.checked_add(1),
                "pass epoch must be last_completed_epoch + 1"
            );
            let max_id: i64 = sqlx::query_scalar("SELECT COALESCE(MAX(id), 0) FROM irises")
                .fetch_one(&mut *self.connection)
                .await?;
            ensure!(max_id < i64::MAX, "iris id range exceeds cursor bounds");
            sqlx::query("SELECT begin_rerand_pass($1, $2, $3)")
                .bind(epoch)
                .bind(max_id)
                .bind(seed_commitment.as_slice())
                .fetch_optional(&mut *self.connection)
                .await?;
            let updated = load_state(&mut self.connection).await?;
            ensure!(
                same_identity(&updated, &self.state),
                "store identity changed"
            );
            self.state = updated;
        }
        Ok((
            self.state.next_id.expect("active state has cursor"),
            self.state.max_id.expect("active state has maximum"),
        ))
    }

    pub async fn fetch_rows(&mut self, start: i64, end: i64) -> Result<Vec<RerandSourceRow>> {
        ensure!(start >= 1 && end >= start, "invalid rerandomization range");
        Ok(sqlx::query_as(
            "SELECT id, version_id, left_code, left_mask, right_code, right_mask, rerand_epoch, \
                    semantic_id::text \
             FROM irises WHERE id >= $1 AND id < $2 ORDER BY id",
        )
        .bind(start)
        .bind(end)
        .fetch_all(&mut *self.connection)
        .await?)
    }

    pub async fn apply(&mut self, updates: &[RerandRowUpdate], to_epoch: u32) -> Result<u64> {
        ensure!(
            to_epoch > 0 && to_epoch <= i32::MAX as u32,
            "invalid target epoch"
        );
        let mut ids_seen = HashSet::with_capacity(updates.len());
        for update in updates {
            ensure!(ids_seen.insert(update.id), "duplicate row {}", update.id);
            ensure!(
                update.from_epoch >= 0 && update.from_epoch < to_epoch as i32,
                "non-forward retarget for row {}",
                update.id
            );
        }
        if updates.is_empty() {
            return Ok(0);
        }
        let ids: Vec<_> = updates.iter().map(|u| u.id).collect();
        let versions: Vec<_> = updates.iter().map(|u| u.expected_version_id).collect();
        let semantic_ids: Vec<_> = updates
            .iter()
            .map(|u| u.expected_semantic_id.clone())
            .collect();
        let epochs: Vec<_> = updates.iter().map(|u| u.from_epoch).collect();
        let lc: Vec<&[u8]> = updates.iter().map(|u| cast_slice(&u.left_code)).collect();
        let lm: Vec<&[u8]> = updates.iter().map(|u| cast_slice(&u.left_mask)).collect();
        let rc: Vec<&[u8]> = updates.iter().map(|u| cast_slice(&u.right_code)).collect();
        let rm: Vec<&[u8]> = updates.iter().map(|u| cast_slice(&u.right_mask)).collect();
        let updated: i64 =
            sqlx::query_scalar("SELECT apply_rerand_updates($1, $2, $3, $4, $5, $6, $7, $8, $9)")
                .bind(ids)
                .bind(versions)
                .bind(semantic_ids)
                .bind(epochs)
                .bind(lc)
                .bind(lm)
                .bind(rc)
                .bind(rm)
                .bind(to_epoch as i32)
                .fetch_one(&mut *self.connection)
                .await?;
        Ok(updated.try_into()?)
    }

    pub async fn advance(&mut self, next_id: i64) -> Result<()> {
        let epoch = self
            .state
            .active_epoch
            .ok_or_else(|| eyre!("no active pass"))?;
        sqlx::query("SELECT advance_rerand_pass($1, $2)")
            .bind(epoch as i32)
            .bind(next_id)
            .fetch_optional(&mut *self.connection)
            .await?;
        self.state.next_id = Some(next_id);
        Ok(())
    }

    pub async fn complete(mut self) -> Result<RerandStoreState> {
        let epoch = self
            .state
            .active_epoch
            .ok_or_else(|| eyre!("no active pass"))?;
        sqlx::query("SELECT complete_rerand_pass($1)")
            .bind(epoch as i32)
            .fetch_optional(&mut *self.connection)
            .await?;
        let state = load_state(&mut self.connection).await?;
        ensure!(same_identity(&state, &self.state), "store identity changed");
        let unlocked: bool = sqlx::query_scalar("SELECT unlock_rerand_pass_lock()")
            .fetch_one(&mut *self.connection)
            .await?;
        ensure!(unlocked, "rerandomization advisory lock was lost");
        self.connection.close().await?;
        Ok(state)
    }
}

async fn load_state(connection: &mut PoolConnection<Postgres>) -> Result<RerandStoreState> {
    let row = sqlx::query_as::<
        _,
        (
            Option<String>,
            Option<String>,
            Option<String>,
            Option<i16>,
            Option<String>,
            Option<String>,
            i32,
            Option<Vec<u8>>,
            Option<i32>,
            Option<Vec<u8>>,
            Option<i64>,
            Option<i64>,
        ),
    >("SELECT * FROM get_rerand_store_state()")
    .fetch_one(&mut **connection)
    .await?;
    let store_id = row
        .0
        .ok_or_else(|| eyre!("store identity is uninitialized"))?;
    Ok(RerandStoreState {
        store_id,
        environment: row.1.ok_or_else(|| eyre!("environment is uninitialized"))?,
        coordination_id: row
            .2
            .ok_or_else(|| eyre!("coordination ID is uninitialized"))?,
        party_id: row
            .3
            .ok_or_else(|| eyre!("party is uninitialized"))?
            .try_into()?,
        store_kind: row.4.ok_or_else(|| eyre!("store kind is uninitialized"))?,
        writer_role: row
            .5
            .ok_or_else(|| eyre!("rerandomization writer is uninitialized"))?,
        last_completed_epoch: row.6.try_into()?,
        last_seed_commitment: parse_commitment(row.7)?,
        active_epoch: row.8.map(u32::try_from).transpose()?,
        active_seed_commitment: parse_commitment(row.9)?,
        next_id: row.10,
        max_id: row.11,
    })
}

fn parse_commitment(value: Option<Vec<u8>>) -> Result<Option<[u8; 32]>> {
    value
        .map(|bytes| {
            bytes
                .try_into()
                .map_err(|_| eyre!("invalid seed commitment length"))
        })
        .transpose()
}

fn same_identity(left: &RerandStoreState, right: &RerandStoreState) -> bool {
    left.store_id == right.store_id
        && left.environment == right.environment
        && left.coordination_id == right.coordination_id
        && left.party_id == right.party_id
        && left.store_kind == right.store_kind
        && left.writer_role == right.writer_role
}

/// Seeds required to normalize authoritative and cached rows to raw epoch 0.
#[derive(Clone)]
pub struct RerandContext {
    party_id: usize,
    seeds: HashMap<u32, EpochSeed>,
}

impl RerandContext {
    pub fn serving_epoch(&self) -> u32 {
        0
    }

    pub fn contains_epoch(&self, epoch: u32) -> bool {
        epoch == 0 || self.seeds.contains_key(&epoch)
    }

    pub fn validate_coverage(&self, inventory: &[(i32, i64)]) -> Result<()> {
        for (epoch, count) in inventory {
            ensure!(*epoch >= 0, "negative row epoch {epoch}");
            ensure!(
                self.contains_epoch(*epoch as u32),
                "missing seed for epoch {epoch}, referenced by {count} rows"
            );
        }
        Ok(())
    }

    fn key(&self, epoch: u32) -> Result<EpochKey<'_>> {
        if epoch == 0 {
            return Ok(EpochKey::new(0, None));
        }
        let seed = self
            .seeds
            .get(&epoch)
            .ok_or_else(|| eyre!("missing seed for rerandomization epoch {epoch}"))?;
        Ok(EpochKey::new(epoch as i32, Some(seed)))
    }

    pub fn normalize_db_iris(&self, iris: &mut DbStoredIris) -> Result<()> {
        ensure!(iris.rerand_epoch >= 0, "negative row epoch");
        let epoch = iris.rerand_epoch as u32;
        if epoch == 0 {
            return Ok(());
        }
        let mut lc = bytes_to_u16(&iris.left_code)?;
        let mut lm = bytes_to_u16(&iris.left_mask)?;
        let mut rc = bytes_to_u16(&iris.right_code)?;
        let mut rm = bytes_to_u16(&iris.right_mask)?;
        retarget_shares(
            self.party_id,
            iris.id,
            self.key(epoch)?,
            EpochKey::new(0, None),
            &mut lc,
            &mut lm,
            &mut rc,
            &mut rm,
        )?;
        iris.left_code = u16_to_bytes(&lc);
        iris.left_mask = u16_to_bytes(&lm);
        iris.right_code = u16_to_bytes(&rc);
        iris.right_mask = u16_to_bytes(&rm);
        iris.rerand_epoch = 0;
        Ok(())
    }

    pub fn normalize_s3_iris(&self, iris: &mut S3StoredIris) -> Result<()> {
        ensure!(iris.rerand_epoch >= 0, "negative cached row epoch");
        let epoch = iris.rerand_epoch as u32;
        if epoch == 0 {
            return Ok(());
        }
        let mut lc = limbs_to_u16(&iris.left_code_odd, &iris.left_code_even)?;
        let mut lm = limbs_to_u16(&iris.left_mask_odd, &iris.left_mask_even)?;
        let mut rc = limbs_to_u16(&iris.right_code_odd, &iris.right_code_even)?;
        let mut rm = limbs_to_u16(&iris.right_mask_odd, &iris.right_mask_even)?;
        retarget_shares(
            self.party_id,
            iris.id,
            self.key(epoch)?,
            EpochKey::new(0, None),
            &mut lc,
            &mut lm,
            &mut rc,
            &mut rm,
        )?;
        u16_to_limbs(&lc, &mut iris.left_code_odd, &mut iris.left_code_even)?;
        u16_to_limbs(&lm, &mut iris.left_mask_odd, &mut iris.left_mask_even)?;
        u16_to_limbs(&rc, &mut iris.right_code_odd, &mut iris.right_code_even)?;
        u16_to_limbs(&rm, &mut iris.right_mask_odd, &mut iris.right_mask_even)?;
        iris.rerand_epoch = 0;
        Ok(())
    }
}

pub fn build_epoch_zero_rerand_context(
    party_id: usize,
    seeds: HashMap<u32, EpochSeed>,
    inventory: &[(i32, i64)],
) -> Result<RerandContext> {
    ensure!(party_id < 3, "party id must be 0, 1, or 2");
    ensure!(!seeds.contains_key(&0), "epoch zero must not have a seed");
    let context = RerandContext { party_id, seeds };
    context.validate_coverage(inventory)?;
    Ok(context)
}

fn bytes_to_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    ensure!(bytes.len() % 2 == 0, "odd share byte length");
    Ok(bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

fn u16_to_bytes(coefs: &[u16]) -> Vec<u8> {
    coefs.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn limbs_to_u16(low: &[u8], high: &[u8]) -> Result<Vec<u16>> {
    ensure!(low.len() == high.len(), "S3 limb planes differ in length");
    Ok(low
        .iter()
        .zip(high)
        .map(|(lo, hi)| u16::from_le_bytes([lo ^ 0x80, hi ^ 0x80]))
        .collect())
}

fn u16_to_limbs(coefs: &[u16], low: &mut [u8], high: &mut [u8]) -> Result<()> {
    ensure!(
        coefs.len() == low.len() && low.len() == high.len(),
        "S3 limb size mismatch"
    );
    for ((value, low), high) in coefs.iter().zip(low).zip(high) {
        let bytes = value.to_le_bytes();
        *low = bytes[0] ^ 0x80;
        *high = bytes[1] ^ 0x80;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

    #[test]
    fn normalizes_database_and_cache_rows_to_epoch_zero() {
        let seed = [9; 32];
        let mut lc = vec![0; IRIS_CODE_LENGTH];
        let mut lm = vec![0; MASK_CODE_LENGTH];
        let mut rc = vec![0; IRIS_CODE_LENGTH];
        let mut rm = vec![0; MASK_CODE_LENGTH];
        retarget_shares(
            1,
            42,
            EpochKey::new(0, None),
            EpochKey::new(7, Some(&seed)),
            &mut lc,
            &mut lm,
            &mut rc,
            &mut rm,
        )
        .unwrap();
        let context =
            build_epoch_zero_rerand_context(1, HashMap::from([(7, seed)]), &[(7, 1)]).unwrap();

        let mut database = DbStoredIris::new_at_epoch(
            42,
            3,
            u16_to_bytes(&lc),
            u16_to_bytes(&lm),
            u16_to_bytes(&rc),
            u16_to_bytes(&rm),
            7,
        );
        context.normalize_db_iris(&mut database).unwrap();
        assert_eq!(database.rerand_epoch(), 0);
        assert!(database.left_code().iter().all(|value| *value == 0));

        let planes = |coefs: &[u16]| {
            let mut low = vec![0; coefs.len()];
            let mut high = vec![0; coefs.len()];
            u16_to_limbs(coefs, &mut low, &mut high).unwrap();
            (low, high)
        };
        let (lc_low, lc_high) = planes(&lc);
        let (lm_low, lm_high) = planes(&lm);
        let (rc_low, rc_high) = planes(&rc);
        let (rm_low, rm_high) = planes(&rm);
        let mut cached = S3StoredIris {
            id: 42,
            left_code_even: lc_high,
            left_code_odd: lc_low,
            left_mask_even: lm_high,
            left_mask_odd: lm_low,
            right_code_even: rc_high,
            right_code_odd: rc_low,
            right_mask_even: rm_high,
            right_mask_odd: rm_low,
            version_id: 3,
            rerand_epoch: 7,
            semantic_id: Some([1; 16]),
        };
        context.normalize_s3_iris(&mut cached).unwrap();
        assert_eq!(cached.rerand_epoch(), 0);
        assert!(limbs_to_u16(&cached.left_code_odd, &cached.left_code_even)
            .unwrap()
            .iter()
            .all(|value| *value == 0));
    }

    #[test]
    fn context_rejects_missing_seed() {
        assert!(build_epoch_zero_rerand_context(0, HashMap::new(), &[(4, 9)]).is_err());
    }

    #[cfg(feature = "db_dependent")]
    #[tokio::test]
    async fn database_enforces_identity_lock_cas_and_semantic_reset() -> Result<()> {
        use crate::test_utils::{cleanup, temporary_name, test_db_url};
        use crate::Store;
        use iris_mpc_common::postgres::{AccessMode, PostgresClient};

        let schema = temporary_name();
        let postgres = PostgresClient::new(&test_db_url()?, &schema, AccessMode::ReadWrite).await?;
        let store = Store::new(&postgres).await?;
        let role = format!("rerand-test-{}", rand::random::<u32>());
        let privileged_role = format!("rerand_privileged_test_{}", rand::random::<u32>());
        let inherited_role = format!("rerand_inherited_test_{}", rand::random::<u32>());
        sqlx::query(&format!("CREATE ROLE \"{role}\" LOGIN"))
            .execute(&store.pool)
            .await?;
        sqlx::query(&format!(
            "CREATE ROLE \"{privileged_role}\" LOGIN CREATEROLE"
        ))
        .execute(&store.pool)
        .await?;
        sqlx::query(&format!("CREATE ROLE \"{inherited_role}\" LOGIN"))
            .execute(&store.pool)
            .await?;
        sqlx::query(&format!("GRANT pg_read_all_data TO \"{inherited_role}\""))
            .execute(&store.pool)
            .await?;

        let result: Result<()> = async {
            sqlx::query(&format!(
                "GRANT USAGE ON SCHEMA \"{schema}\" TO \"{role}\""
            ))
            .execute(&store.pool)
            .await?;
            sqlx::query(&format!("GRANT SELECT ON irises TO \"{role}\""))
                .execute(&store.pool)
                .await?;
            assert!(
                sqlx::query("SELECT initialize_rerand_store($1, $2, $3, $4, $5, $6::name)")
                    .bind("test-store")
                    .bind("test")
                    .bind("generation-test")
                    .bind(1_i16)
                    .bind("gpu")
                    .bind(&privileged_role)
                    .execute(&store.pool)
                    .await
                    .is_err()
            );
            assert!(
                sqlx::query("SELECT initialize_rerand_store($1, $2, $3, $4, $5, $6::name)")
                    .bind("test-store")
                    .bind("test")
                    .bind("generation-test")
                    .bind(1_i16)
                    .bind("gpu")
                    .bind(&inherited_role)
                    .execute(&store.pool)
                    .await
                    .is_err()
            );
            sqlx::query("SELECT initialize_rerand_store($1, $2, $3, $4, $5, $6::name)")
                .bind("test-store")
                .bind("test")
                .bind("generation-test")
                .bind(1_i16)
                .bind("gpu")
                .bind(&role)
                .execute(&store.pool)
                .await?;
            let reported_writer: String =
                sqlx::query_scalar("SELECT writer_role FROM get_rerand_store_state()")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                reported_writer == role,
                "writer role was returned as a quoted regrole expression"
            );
            let always_trigger_count: i64 = sqlx::query_scalar(
                "SELECT COUNT(*) FROM pg_catalog.pg_trigger \
                 WHERE tgname IN (\
                     'protect_rerand_control_trigger', \
                     'update_iris_metadata_trigger', \
                     'protect_iris_truncate_trigger'\
                 ) AND tgenabled = 'A'",
            )
            .fetch_one(&store.pool)
            .await?;
            ensure!(
                always_trigger_count == 3,
                "rerandomization invariant triggers are not ENABLE ALWAYS"
            );
            let function_owner_matches_table: bool = sqlx::query_scalar(
                "SELECT p.proowner = i.relowner \
                 FROM pg_catalog.pg_proc p \
                 CROSS JOIN pg_catalog.pg_class i \
                 WHERE p.oid = 'apply_rerand_updates(bigint[],smallint[],text[],integer[],bytea[],bytea[],bytea[],bytea[],integer)'::regprocedure \
                   AND i.oid = 'irises'::regclass",
            )
            .fetch_one(&store.pool)
            .await?;
            ensure!(
                function_owner_matches_table,
                "rerandomization function owner differs from the irises owner"
            );
            sqlx::query(
                "INSERT INTO irises \
                 (id, left_code, left_mask, right_code, right_mask, semantic_id) \
                 VALUES (1, '\\x00', '\\x00', '\\x00', '\\x00', \
                         '00000000-0000-0000-0000-000000000001')",
            )
            .execute(&store.pool)
            .await?;
            let inserted_semantic_id: String =
                sqlx::query_scalar("SELECT semantic_id::text FROM irises WHERE id = 1")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                inserted_semantic_id != "00000000-0000-0000-0000-000000000001",
                "insert accepted a caller-selected semantic ID"
            );
            sqlx::query(
                "INSERT INTO irises (id, left_code, left_mask, right_code, right_mask) \
                 VALUES (2, '\\x00', '\\x00', '\\x00', '\\x00')",
            )
            .execute(&store.pool)
            .await?;
            sqlx::query("ALTER TABLE irises DISABLE TRIGGER update_iris_metadata_trigger")
                .execute(&store.pool)
                .await?;
            sqlx::query("UPDATE irises SET semantic_id = NULL WHERE id = 2")
                .execute(&store.pool)
                .await?;
            sqlx::query(
                "ALTER TABLE irises ENABLE ALWAYS TRIGGER update_iris_metadata_trigger",
            )
                .execute(&store.pool)
                .await?;
            let legacy_semantic_id: Option<String> =
                sqlx::query_scalar("SELECT semantic_id::text FROM irises WHERE id = 2")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                legacy_semantic_id.is_none(),
                "failed to create a legacy bootstrap row"
            );
            sqlx::query(&format!(
                "GRANT INSERT, UPDATE, DELETE, TRUNCATE ON irises TO \"{role}\""
            ))
            .execute(&store.pool)
            .await?;

            let mut writer = store.pool.acquire().await?;
            writer.close_on_drop();
            sqlx::query(&format!("SET SESSION AUTHORIZATION \"{role}\""))
                .execute(&mut *writer)
                .await?;
            sqlx::query(&format!("SET search_path TO pg_temp, \"{schema}\", public"))
                .execute(&mut *writer)
                .await?;
            sqlx::query("CREATE TEMP TABLE rerand_control (singleton boolean)")
                .execute(&mut *writer)
                .await?;
            assert!(
                sqlx::query_scalar::<_, bool>(&format!(
                    "SELECT \"{schema}\".try_rerand_pass_lock()"
                ))
                .fetch_one(&mut *writer)
                .await?
            );
            assert!(sqlx::query(&format!(
                "SELECT \"{schema}\".begin_rerand_pass(\
                 1, NULL::bigint, decode(repeat('01', 32), 'hex'))"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            assert!(sqlx::query(&format!(
                "SELECT \"{schema}\".complete_rerand_pass(NULL::integer)"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            sqlx::query(&format!(
                "SELECT \"{schema}\".begin_rerand_pass(1, 2, decode(repeat('01', 32), 'hex'))"
            ))
            .execute(&mut *writer)
            .await?;
            assert!(sqlx::query(&format!(
                "SELECT \"{schema}\".advance_rerand_pass(1, NULL::bigint)"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            assert!(sqlx::query(&format!(
                "UPDATE \"{schema}\".irises \
                 SET left_code = '\\xfe', rerand_epoch = 1 WHERE id = 1"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            assert!(sqlx::query(&format!(
                "UPDATE \"{schema}\".irises SET left_code = '\\xff' WHERE id = 1"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            assert!(sqlx::query(&format!(
                "INSERT INTO \"{schema}\".irises \
                 (id, left_code, left_mask, right_code, right_mask) \
                 VALUES (3, '\\x00', '\\x00', '\\x00', '\\x00')"
            ))
            .execute(&mut *writer)
            .await
            .is_err());
            assert!(
                sqlx::query(&format!("DELETE FROM \"{schema}\".irises WHERE id = 1"))
                    .execute(&mut *writer)
                    .await
                    .is_err()
            );
            assert!(sqlx::query(&format!("TRUNCATE \"{schema}\".irises"))
                .execute(&mut *writer)
                .await
                .is_err());

            sqlx::query("DELETE FROM irises WHERE id = 1")
                .execute(&store.pool)
                .await?;
            sqlx::query(
                "INSERT INTO irises \
                 (id, left_code, left_mask, right_code, right_mask) \
                 VALUES (1, '\\x7f', '\\x7f', '\\x7f', '\\x7f')",
            )
            .execute(&store.pool)
            .await?;
            let replacement_semantic_id: String =
                sqlx::query_scalar("SELECT semantic_id::text FROM irises WHERE id = 1")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                replacement_semantic_id != inserted_semantic_id,
                "delete/reinsert reused the old semantic identity"
            );

            let stale_applied: i64 = sqlx::query_scalar(&format!(
                "SELECT \"{schema}\".apply_rerand_updates(\
                 ARRAY[1]::bigint[], ARRAY[0]::smallint[], \
                 ARRAY['{inserted_semantic_id}']::text[], ARRAY[0]::integer[], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], 1)"
            ))
            .fetch_one(&mut *writer)
            .await?;
            ensure!(
                stale_applied == 0,
                "delayed CAS overwrote a replacement row"
            );
            let replacement_before_apply: (Vec<u8>, i32, String) = sqlx::query_as(
                "SELECT left_code, rerand_epoch, semantic_id::text \
                 FROM irises WHERE id = 1",
            )
            .fetch_one(&store.pool)
            .await?;
            ensure!(
                replacement_before_apply
                    == (vec![0x7f], 0, replacement_semantic_id.clone()),
                "stale CAS changed the replacement row"
            );

            let applied: i64 = sqlx::query_scalar(&format!(
                "SELECT \"{schema}\".apply_rerand_updates(\
                 ARRAY[1]::bigint[], ARRAY[0]::smallint[], \
                 ARRAY['{replacement_semantic_id}']::text[], ARRAY[0]::integer[], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], 1)"
            ))
            .fetch_one(&mut *writer)
            .await?;
            ensure!(applied == 1, "guarded rerandomization did not apply");
            sqlx::query(&format!("SELECT \"{schema}\".advance_rerand_pass(1, 3)"))
                .execute(&mut *writer)
                .await?;
            assert!(
                sqlx::query(&format!("SELECT \"{schema}\".complete_rerand_pass(1)"))
                    .execute(&mut *writer)
                    .await
                    .is_err()
            );
            let backfilled: i64 = sqlx::query_scalar(&format!(
                "SELECT \"{schema}\".apply_rerand_updates(\
                 ARRAY[2]::bigint[], ARRAY[0]::smallint[], \
                 ARRAY[NULL]::text[], ARRAY[0]::integer[], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], \
                 ARRAY['\\x01'::bytea], ARRAY['\\x01'::bytea], 1)"
            ))
            .fetch_one(&mut *writer)
            .await?;
            ensure!(backfilled == 1, "legacy semantic ID backfill did not apply");
            sqlx::query(&format!("SELECT \"{schema}\".complete_rerand_pass(1)"))
                .execute(&mut *writer)
                .await?;
            assert!(
                sqlx::query_scalar::<_, bool>(&format!(
                    "SELECT \"{schema}\".unlock_rerand_pass_lock()"
                ))
                .fetch_one(&mut *writer)
                .await?
            );
            sqlx::query("RESET SESSION AUTHORIZATION")
                .execute(&mut *writer)
                .await?;
            writer.close().await?;

            let rerand_semantic_id: String =
                sqlx::query_scalar("SELECT semantic_id::text FROM irises WHERE id = 1")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                rerand_semantic_id == replacement_semantic_id,
                "rerandomization changed the semantic ID"
            );
            let backfilled_semantic_id: Option<String> =
                sqlx::query_scalar("SELECT semantic_id::text FROM irises WHERE id = 2")
                    .fetch_one(&store.pool)
                    .await?;
            ensure!(
                backfilled_semantic_id.is_some(),
                "rerandomization did not backfill the legacy semantic ID"
            );
            assert!(
                sqlx::query("UPDATE irises SET semantic_id = gen_random_uuid() WHERE id = 1")
                    .execute(&store.pool)
                    .await
                    .is_err()
            );
            sqlx::query("UPDATE irises SET left_code = '\\x02' WHERE id = 1")
                .execute(&store.pool)
                .await?;
            let metadata: (i16, i32, String) = sqlx::query_as(
                "SELECT version_id, rerand_epoch, semantic_id::text \
                 FROM irises WHERE id = 1",
            )
            .fetch_one(&store.pool)
            .await?;
            ensure!(
                metadata.0 == 1 && metadata.1 == 0 && metadata.2 != replacement_semantic_id,
                "semantic write metadata is not truthful"
            );

            sqlx::query("UPDATE irises SET left_code = '\\x02' WHERE id = 2")
                .execute(&store.pool)
                .await?;
            sqlx::raw_sql(include_str!(
                "../../migrations/20260712000001_rerandomization.down.sql"
            ))
            .execute(&store.pool)
            .await?;
            let preserved_schema_usage: bool = sqlx::query_scalar(
                "SELECT pg_catalog.has_schema_privilege($1, $2, 'USAGE')",
            )
            .bind(&role)
            .bind(&schema)
            .fetch_one(&store.pool)
            .await?;
            let qualified_irises = format!("\"{schema}\".irises");
            let preserved_iris_select: bool = sqlx::query_scalar(
                "SELECT pg_catalog.has_table_privilege($1, $2, 'SELECT')",
            )
            .bind(&role)
            .bind(&qualified_irises)
            .fetch_one(&store.pool)
            .await?;
            ensure!(
                preserved_schema_usage && preserved_iris_select,
                "rollback revoked writer privileges that predated rerandomization"
            );
            Ok(())
        }
        .await;

        cleanup(&postgres, &schema).await?;
        sqlx::query(&format!("DROP ROLE \"{role}\""))
            .execute(&store.pool)
            .await?;
        sqlx::query(&format!("DROP ROLE \"{privileged_role}\""))
            .execute(&store.pool)
            .await?;
        sqlx::query(&format!(
            "REVOKE pg_read_all_data FROM \"{inherited_role}\""
        ))
        .execute(&store.pool)
        .await?;
        sqlx::query(&format!("DROP ROLE \"{inherited_role}\""))
            .execute(&store.pool)
            .await?;
        result
    }
}
