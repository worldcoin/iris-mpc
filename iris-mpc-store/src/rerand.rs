use std::cmp::Ordering;
use std::time::Duration;

use eyre::Result;
use iris_mpc_common::helpers::sync::{RerandSyncState, SyncResult};
use sqlx::PgPool;

pub const RERAND_APPLY_LOCK: i64 = 0x5245_5241_4E44;
pub const RERAND_MODIFY_LOCK: i64 = 0x5245_4D4F_4446;

pub struct StagingIrisEntry {
    pub epoch: i32,
    pub id: i64,
    pub chunk_id: i32,
    pub left_code: Vec<u8>,
    pub left_mask: Vec<u8>,
    pub right_code: Vec<u8>,
    pub right_mask: Vec<u8>,
    pub original_version_id: i16,
    pub rerand_epoch: i32,
}

#[derive(sqlx::FromRow, Debug, Clone)]
pub struct RerandProgress {
    pub epoch: i32,
    pub chunk_id: i32,
    pub staging_written: bool,
    pub all_confirmed: bool,
    pub live_applied: bool,
}

pub fn staging_schema_name(live_schema: &str) -> String {
    format!("{}_rerand_staging", live_schema)
}

fn validate_identifier(name: &str) -> Result<()> {
    if name.is_empty() {
        eyre::bail!("SQL identifier must not be empty");
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        eyre::bail!(
            "SQL identifier contains invalid characters (only ASCII alphanumeric and _ allowed): {:?}",
            name
        );
    }
    Ok(())
}

/// Delete any partial staging data for a chunk before (re-)staging.
/// Ensures all rows come from one read pass, preventing mixed-snapshot
/// version_ids after a crash-and-retry.
pub async fn delete_staging_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<u64> {
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND chunk_id = $2"#,
        staging_schema,
    );
    let result = sqlx::query(&sql)
        .bind(epoch)
        .bind(chunk_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Return the (id, original_version_id) pairs from staging for a chunk.
pub async fn get_staging_version_map(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<Vec<(i64, i16)>> {
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"SELECT id, original_version_id FROM "{}".irises WHERE epoch = $1 AND chunk_id = $2 ORDER BY id"#,
        staging_schema,
    );
    let rows: Vec<(i64, i16)> = sqlx::query_as(&sql)
        .bind(epoch)
        .bind(chunk_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

/// Return IDs where the staging original_version_id no longer matches the
/// live version_id (modifications landed after staging).
pub async fn get_locally_divergent_ids(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<Vec<i64>> {
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"
        SELECT s.id FROM "{}".irises s
        JOIN irises ON irises.id = s.id
        WHERE s.epoch = $1 AND s.chunk_id = $2
          AND irises.version_id != s.original_version_id
        "#,
        staging_schema,
    );
    let rows: Vec<(i64,)> = sqlx::query_as(&sql)
        .bind(epoch)
        .bind(chunk_id)
        .fetch_all(pool)
        .await?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}

async fn get_locally_divergent_ids_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<Vec<i64>> {
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"
        SELECT s.id FROM "{}".irises s
        JOIN irises ON irises.id = s.id
        WHERE s.epoch = $1 AND s.chunk_id = $2
          AND irises.version_id != s.original_version_id
        "#,
        staging_schema,
    );
    let rows: Vec<(i64,)> = sqlx::query_as(&sql)
        .bind(epoch)
        .bind(chunk_id)
        .fetch_all(&mut **tx)
        .await?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}

/// Delete specific IDs from a staging chunk.
pub async fn delete_staging_ids(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    ids: &[i64],
) -> Result<u64> {
    if ids.is_empty() {
        return Ok(0);
    }
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND id = ANY($2)"#,
        staging_schema,
    );
    let result = sqlx::query(&sql)
        .bind(epoch)
        .bind(ids)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

async fn delete_staging_ids_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    staging_schema: &str,
    epoch: i32,
    ids: &[i64],
) -> Result<u64> {
    if ids.is_empty() {
        return Ok(0);
    }
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND id = ANY($2)"#,
        staging_schema,
    );
    let result = sqlx::query(&sql)
        .bind(epoch)
        .bind(ids)
        .execute(&mut **tx)
        .await?;
    Ok(result.rows_affected())
}

pub async fn insert_staging_irises(
    pool: &PgPool,
    staging_schema: &str,
    entries: &[StagingIrisEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    validate_identifier(staging_schema)?;

    let table = format!("\"{}\".irises", staging_schema);
    let header = format!(
        "INSERT INTO {} (epoch, id, chunk_id, left_code, left_mask, right_code, right_mask, original_version_id, rerand_epoch)",
        table
    );

    let mut qb = sqlx::QueryBuilder::new(header);
    qb.push_values(entries, |mut b, e| {
        b.push_bind(e.epoch);
        b.push_bind(e.id);
        b.push_bind(e.chunk_id);
        b.push_bind(&e.left_code);
        b.push_bind(&e.left_mask);
        b.push_bind(&e.right_code);
        b.push_bind(&e.right_mask);
        b.push_bind(e.original_version_id);
        b.push_bind(e.rerand_epoch);
    });

    qb.push(" ON CONFLICT (epoch, id) DO NOTHING");
    qb.build().execute(pool).await?;
    Ok(())
}

/// Apply a confirmed staging chunk to the live DB.
///
/// Within a single transaction:
///   1. Acquire `pg_advisory_xact_lock(RERAND_APPLY_LOCK)` (released
///      automatically on commit/rollback/connection-drop).
///   2. UPDATE live irises from staging (optimistic lock on version_id)
///   3. DELETE staging rows for this chunk
///   4. Mark live_applied in rerand_progress
pub async fn apply_staging_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<u64> {
    validate_identifier(staging_schema)?;
    let mut tx = pool.begin().await?;
    let rows_updated = apply_staging_chunk_in_tx(&mut tx, staging_schema, epoch, chunk_id).await?;
    tx.commit().await?;
    Ok(rows_updated)
}

async fn apply_staging_chunk_in_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<u64> {
    validate_identifier(staging_schema)?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(RERAND_APPLY_LOCK)
        .execute(&mut **tx)
        .await?;

    let update_sql = format!(
        r#"
        UPDATE irises SET
            left_code    = staging.left_code,
            left_mask    = staging.left_mask,
            right_code   = staging.right_code,
            right_mask   = staging.right_mask,
            rerand_epoch = staging.rerand_epoch
        FROM "{}".irises AS staging
        WHERE irises.id = staging.id
          AND staging.epoch = $1
          AND staging.chunk_id = $2
          AND irises.version_id = staging.original_version_id
        "#,
        staging_schema,
    );
    let result = sqlx::query(&update_sql)
        .bind(epoch)
        .bind(chunk_id)
        .execute(&mut **tx)
        .await?;
    let rows_updated = result.rows_affected();

    let delete_sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND chunk_id = $2"#,
        staging_schema,
    );
    sqlx::query(&delete_sql)
        .bind(epoch)
        .bind(chunk_id)
        .execute(&mut **tx)
        .await?;

    sqlx::query(
        "UPDATE rerand_progress SET live_applied = TRUE WHERE epoch = $1 AND chunk_id = $2",
    )
    .bind(epoch)
    .bind(chunk_id)
    .execute(&mut **tx)
    .await?;

    Ok(rows_updated)
}

/// Apply a chunk under the modification fence in one transaction.
///
/// Transaction scope:
///   1. Acquire `pg_advisory_xact_lock(RERAND_MODIFY_LOCK)`
///   2. Compute local diverged IDs
///   3. Prune union(cross_party_diverged, local_diverged) from staging
///   4. Apply staging chunk to live (`RERAND_APPLY_LOCK` is acquired inside)
///   5. Commit (releasing both transaction locks)
pub async fn fenced_apply_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
    cross_party_divergent: Vec<i64>,
) -> Result<(u64, usize)> {
    validate_identifier(staging_schema)?;
    let mut tx = pool.begin().await?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(RERAND_MODIFY_LOCK)
        .execute(&mut *tx)
        .await?;

    let local_divergent =
        get_locally_divergent_ids_tx(&mut tx, staging_schema, epoch, chunk_id).await?;

    let mut skip_ids = cross_party_divergent;
    skip_ids.extend(&local_divergent);
    skip_ids.sort_unstable();
    skip_ids.dedup();
    let skip_count = skip_ids.len();

    if !skip_ids.is_empty() {
        delete_staging_ids_tx(&mut tx, staging_schema, epoch, &skip_ids).await?;
    }

    let rows = apply_staging_chunk_in_tx(&mut tx, staging_schema, epoch, chunk_id).await?;
    tx.commit().await?;
    Ok((rows, skip_count))
}

pub async fn upsert_rerand_progress(pool: &PgPool, epoch: i32, chunk_id: i32) -> Result<()> {
    sqlx::query(
        r#"
        INSERT INTO rerand_progress (epoch, chunk_id)
        VALUES ($1, $2)
        ON CONFLICT (epoch, chunk_id) DO NOTHING
        "#,
    )
    .bind(epoch)
    .bind(chunk_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_staging_written(pool: &PgPool, epoch: i32, chunk_id: i32) -> Result<()> {
    sqlx::query(
        "UPDATE rerand_progress SET staging_written = TRUE WHERE epoch = $1 AND chunk_id = $2",
    )
    .bind(epoch)
    .bind(chunk_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_all_confirmed(pool: &PgPool, epoch: i32, chunk_id: i32) -> Result<()> {
    sqlx::query(
        "UPDATE rerand_progress SET all_confirmed = TRUE WHERE epoch = $1 AND chunk_id = $2",
    )
    .bind(epoch)
    .bind(chunk_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn get_rerand_progress(
    pool: &PgPool,
    epoch: i32,
    chunk_id: i32,
) -> Result<Option<RerandProgress>> {
    let row = sqlx::query_as::<_, RerandProgress>(
        "SELECT epoch, chunk_id, staging_written, all_confirmed, live_applied FROM rerand_progress WHERE epoch = $1 AND chunk_id = $2",
    )
    .bind(epoch)
    .bind(chunk_id)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// Returns the highest chunk_id where all_confirmed = TRUE for a given epoch,
/// or None if no chunks are confirmed.
pub async fn get_max_confirmed_chunk(pool: &PgPool, epoch: i32) -> Result<Option<i32>> {
    let row: (Option<i32>,) = sqlx::query_as(
        "SELECT MAX(chunk_id) FROM rerand_progress WHERE epoch = $1 AND all_confirmed = TRUE",
    )
    .bind(epoch)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Returns the highest epoch that has any rerand_progress rows.
pub async fn get_current_epoch(pool: &PgPool) -> Result<Option<i32>> {
    let row: (Option<i32>,) = sqlx::query_as("SELECT MAX(epoch) FROM rerand_progress")
        .fetch_one(pool)
        .await?;
    Ok(row.0)
}

// ---------------------------------------------------------------------------
// Shared startup helpers (used by both HNSW and GPU servers)
// ---------------------------------------------------------------------------

/// Build the rerand sync state from the local `rerand_progress` table.
///
/// Returns `Ok(None)` when the `rerand_progress` table does not exist yet
/// (rolling deploy before migration). Returns `Err` for real DB failures
/// so callers can distinguish "not migrated" from "broken".
pub async fn build_rerand_sync_state(pool: &PgPool) -> Result<Option<RerandSyncState>> {
    let epoch = match get_current_epoch(pool).await {
        Ok(e) => e.unwrap_or(0),
        Err(e) => {
            if is_undefined_table(&e) {
                return Ok(None);
            }
            return Err(e);
        }
    };
    let max_confirmed = get_max_confirmed_chunk(pool, epoch).await?.unwrap_or(-1);
    Ok(Some(RerandSyncState {
        epoch,
        max_confirmed_chunk: max_confirmed,
    }))
}

fn is_undefined_table(err: &eyre::Report) -> bool {
    if let Some(db_err) = err.root_cause().downcast_ref::<sqlx::Error>() {
        return is_undefined_table_sqlx(db_err);
    }
    // Also check the direct error (not just root cause).
    format!("{:?}", err).contains("42P01")
}

fn is_undefined_table_sqlx(err: &sqlx::Error) -> bool {
    if let sqlx::Error::Database(pg) = err {
        return pg.code().as_deref() == Some("42P01");
    }
    false
}

/// Check whether all locally confirmed chunks have been applied to live.
///
/// Returns `Ok(true)` when no confirmed-but-unapplied chunks remain,
/// `Ok(true)` when the `rerand_progress` table doesn't exist yet
/// (rolling deploy), and `Err` on real DB failures.
async fn check_pending_chunks_applied(conn: &mut sqlx::PgConnection) -> Result<bool> {
    let pending: (i64,) = match sqlx::query_as(
        "SELECT COUNT(*) FROM rerand_progress \
         WHERE all_confirmed = TRUE AND live_applied = FALSE",
    )
    .fetch_one(&mut *conn)
    .await
    {
        Ok(row) => row,
        Err(e) if is_undefined_table_sqlx(&e) => return Ok(true),
        Err(e) => return Err(e.into()),
    };
    Ok(pending.0 == 0)
}

/// Highest `(epoch, chunk_id)` where `live_applied = TRUE`.
/// Returns `None` when no chunks have been applied yet.
async fn get_applied_watermark(conn: &mut sqlx::PgConnection) -> Result<Option<(i32, i32)>> {
    let row: Option<(i32, i32)> = match sqlx::query_as(
        "SELECT epoch, chunk_id FROM rerand_progress \
         WHERE live_applied = TRUE \
         ORDER BY epoch DESC, chunk_id DESC \
         LIMIT 1",
    )
    .fetch_optional(&mut *conn)
    .await
    {
        Ok(row) => row,
        Err(e) if is_undefined_table_sqlx(&e) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    Ok(row)
}

/// Highest (epoch, max_confirmed_chunk) reported by any peer in the
/// startup snapshot. Returns `None` when no peer has rerand state
/// (pre-migration rolling deploy).
fn peer_rerand_target(sync_result: &SyncResult) -> Option<(i32, i32)> {
    sync_result
        .all_states
        .iter()
        .filter_map(|s| s.rerand_state.as_ref())
        .map(|s| (s.epoch, s.max_confirmed_chunk))
        .max() // lexicographic: epoch first, then chunk
}

/// Returns `Ok(())` if the peer snapshot is within protocol tolerance
/// and `Err` if fatally desynchronized (gap > 1).
fn validate_rerand_sync_inner(sync_result: &SyncResult) -> Result<()> {
    let my_state = match sync_result.my_state.rerand_state.as_ref() {
        Some(s) => s,
        None => return Ok(()),
    };
    let my_epoch = my_state.epoch;
    let my_chunk = my_state.max_confirmed_chunk;

    for s in sync_result
        .all_states
        .iter()
        .filter_map(|s| s.rerand_state.as_ref())
    {
        let epoch_diff = s.epoch - my_epoch;
        match epoch_diff {
            0 => {
                let chunk_diff = s.max_confirmed_chunk - my_chunk;
                if chunk_diff > 1 {
                    eyre::bail!(
                        "Fatal chunk desync: peer confirmed chunk {} but local is at {} \
                         (max possible difference is 1)",
                        s.max_confirmed_chunk,
                        my_chunk
                    );
                }
            }
            1 => {}
            -1 => {}
            _ => {
                eyre::bail!(
                    "Fatal epoch desync: local epoch is {}, but peer is on epoch {}",
                    my_epoch,
                    s.epoch
                );
            }
        }
    }

    Ok(())
}

const RERAND_READY_TIMEOUT: Duration = Duration::from_secs(60);
const RERAND_READY_POLL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
enum StartupReadiness {
    Ready,
    Behind,
    Ahead {
        local_applied: (i32, i32),
        target: (i32, i32),
    },
}

fn classify_startup_readiness_for_target(
    local_applied: (i32, i32),
    target: (i32, i32),
) -> StartupReadiness {
    if local_applied.cmp(&target) == Ordering::Greater {
        return StartupReadiness::Ahead {
            local_applied,
            target,
        };
    }

    if target.1 < 0 {
        // No confirmed chunks exist in target_epoch yet.
        return StartupReadiness::Ready;
    }

    if local_applied == target {
        StartupReadiness::Ready
    } else {
        StartupReadiness::Behind
    }
}

async fn get_startup_readiness(
    conn: &mut sqlx::PgConnection,
    target: Option<(i32, i32)>,
) -> Result<StartupReadiness> {
    if !check_pending_chunks_applied(conn).await? {
        return Ok(StartupReadiness::Behind);
    }

    let Some(target) = target else {
        return Ok(StartupReadiness::Ready);
    };

    let local_applied = get_applied_watermark(conn).await?.unwrap_or((-1, -1));
    Ok(classify_startup_readiness_for_target(local_applied, target))
}

/// Wait for local rerand progress to reach the startup snapshot target,
/// then hold `RERAND_APPLY_LOCK` through DB load.
///
/// The loop is lock-first:
/// 1. acquire `pg_advisory_lock(RERAND_APPLY_LOCK)`,
/// 2. check readiness while applies are frozen,
/// 3. if behind, unlock and retry after a short sleep.
///
/// This avoids startup/apply races without a separate startup-cap table.
pub async fn rerand_validate_and_lock(
    pool: &PgPool,
    sync_result: &SyncResult,
) -> Result<Option<sqlx::pool::PoolConnection<sqlx::Postgres>>> {
    if sync_result.my_state.rerand_state.is_none() {
        tracing::info!("Rerand startup lock: skipped (rerand tables not yet migrated)");
        return Ok(None);
    }

    // One-shot fatal desync check (gap > 1 -> bail).
    validate_rerand_sync_inner(sync_result)?;

    let target = peer_rerand_target(sync_result);
    let deadline = tokio::time::Instant::now() + RERAND_READY_TIMEOUT;

    loop {
        let mut conn = pool.acquire().await?;
        let got_lock: (bool,) = sqlx::query_as("SELECT pg_try_advisory_lock($1)")
            .bind(RERAND_APPLY_LOCK)
            .fetch_one(&mut *conn)
            .await?;
        if !got_lock.0 {
            drop(conn);
            if tokio::time::Instant::now() >= deadline {
                eyre::bail!(
                    "Rerand lock not available after {:?} (target={:?}); \
                     ensure the rerand worker is healthy.",
                    RERAND_READY_TIMEOUT,
                    target
                );
            }
            tokio::time::sleep(RERAND_READY_POLL).await;
            continue;
        }

        let readiness = match get_startup_readiness(&mut conn, target).await {
            Ok(readiness) => readiness,
            Err(e) => {
                let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
                    .bind(RERAND_APPLY_LOCK)
                    .execute(&mut *conn)
                    .await;
                drop(conn);
                return Err(e);
            }
        };

        match readiness {
            StartupReadiness::Ready => return Ok(Some(conn)),
            StartupReadiness::Ahead {
                local_applied,
                target,
            } => {
                let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
                    .bind(RERAND_APPLY_LOCK)
                    .execute(&mut *conn)
                    .await;
                drop(conn);
                eyre::bail!(
                    "Rerand advanced past startup snapshot target: local_applied={:?}, target={:?}. \
                     Restart and retry startup.",
                    local_applied,
                    target,
                );
            }
            StartupReadiness::Behind => {
                let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
                    .bind(RERAND_APPLY_LOCK)
                    .execute(&mut *conn)
                    .await;
                drop(conn);
            }
        }

        if tokio::time::Instant::now() >= deadline {
            eyre::bail!(
                "Rerand not caught up after {:?} (target={:?}); \
                 ensure the rerand worker is running.",
                RERAND_READY_TIMEOUT,
                target
            );
        }

        tracing::info!(
            "Waiting for rerand worker catch-up (target={:?}, {:.0}s left)...",
            target,
            deadline
                .saturating_duration_since(tokio::time::Instant::now())
                .as_secs_f64(),
        );
        tokio::time::sleep(RERAND_READY_POLL).await;
    }
}

/// Release the advisory lock and close the connection.
///
/// Explicit release keeps the lock lifecycle clear in logs and avoids
/// returning a locked connection to the pool.
pub async fn release_rerand_lock(
    lock_conn: Option<sqlx::pool::PoolConnection<sqlx::Postgres>>,
) -> Result<()> {
    if let Some(mut conn) = lock_conn {
        let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
            .bind(RERAND_APPLY_LOCK)
            .execute(&mut *conn)
            .await;
        drop(conn);
        tracing::info!("Rerand advisory lock released after DB load");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris_mpc_common::config::CommonConfig;
    use iris_mpc_common::helpers::sync::SyncState;

    fn dummy_sync_state(epoch: i32, max_confirmed_chunk: i32) -> SyncState {
        SyncState {
            db_len: 100,
            modifications: vec![],
            next_sns_sequence_num: None,
            common_config: CommonConfig::default(),
            rerand_state: Some(RerandSyncState {
                epoch,
                max_confirmed_chunk,
            }),
        }
    }

    #[test]
    fn test_validate_peer_one_chunk_ahead_ok() {
        let p0 = dummy_sync_state(1, 4);
        let p1 = dummy_sync_state(1, 4);
        let p2 = dummy_sync_state(1, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_ok());
    }

    #[test]
    fn test_validate_all_same_ok() {
        let p0 = dummy_sync_state(1, 5);
        let p1 = dummy_sync_state(1, 5);
        let p2 = dummy_sync_state(1, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_ok());
    }

    #[test]
    fn test_validate_peer_epoch_ahead_ok() {
        let p0 = dummy_sync_state(0, 5);
        let p1 = dummy_sync_state(1, 0);
        let p2 = dummy_sync_state(0, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_ok());
    }

    #[test]
    fn test_validate_peer_epoch_behind_ok() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(0, 10);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_ok());
    }

    #[test]
    fn test_validate_fatal_chunk_desync() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(1, 4);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_err());
    }

    #[test]
    fn test_validate_fatal_epoch_desync() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(3, 10);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(validate_rerand_sync_inner(&sync_result).is_err());
    }

    #[test]
    fn test_classify_target_chunk_minus_one_previous_epoch_applied_is_ready() {
        let readiness = classify_startup_readiness_for_target((0, 42), (1, -1));
        assert_eq!(readiness, StartupReadiness::Ready);
    }

    #[test]
    fn test_classify_target_chunk_minus_one_same_epoch_applied_is_ahead() {
        let readiness = classify_startup_readiness_for_target((1, 0), (1, -1));
        assert_eq!(
            readiness,
            StartupReadiness::Ahead {
                local_applied: (1, 0),
                target: (1, -1)
            }
        );
    }

    #[test]
    fn test_classify_target_positive_behind_ready_ahead() {
        assert_eq!(
            classify_startup_readiness_for_target((1, 2), (1, 3)),
            StartupReadiness::Behind
        );
        assert_eq!(
            classify_startup_readiness_for_target((1, 3), (1, 3)),
            StartupReadiness::Ready
        );
        assert_eq!(
            classify_startup_readiness_for_target((1, 4), (1, 3)),
            StartupReadiness::Ahead {
                local_applied: (1, 4),
                target: (1, 3)
            }
        );
    }
}
