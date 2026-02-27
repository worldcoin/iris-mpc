use eyre::Result;
use iris_mpc_common::helpers::sync::{RerandSyncState, SyncResult};
use sqlx::{pool::PoolConnection, PgPool, Postgres};

pub const RERAND_APPLY_LOCK: i64 = 0x5245_5241_4E44;

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

pub async fn ensure_staging_schema(pool: &PgPool, staging_schema: &str) -> Result<()> {
    let create_schema = format!(r#"CREATE SCHEMA IF NOT EXISTS "{}""#, staging_schema);
    sqlx::query(&create_schema).execute(pool).await?;

    let create_table = format!(
        r#"
        CREATE TABLE IF NOT EXISTS "{}".irises (
            epoch               INTEGER NOT NULL,
            id                  BIGINT NOT NULL,
            chunk_id            INTEGER NOT NULL,
            left_code           BYTEA,
            left_mask           BYTEA,
            right_code          BYTEA,
            right_mask          BYTEA,
            original_version_id SMALLINT,
            rerand_epoch        INTEGER,
            PRIMARY KEY (epoch, id)
        )
        "#,
        staging_schema,
    );
    sqlx::query(&create_table).execute(pool).await?;
    Ok(())
}

pub async fn insert_staging_irises(
    pool: &PgPool,
    staging_schema: &str,
    entries: &[StagingIrisEntry],
) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }

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
///   1. UPDATE live irises from staging (optimistic lock on version_id)
///   2. DELETE staging rows for this chunk
///   3. Mark live_applied in rerand_progress
pub async fn apply_staging_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<u64> {
    let mut tx = pool.begin().await?;

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
        .execute(&mut *tx)
        .await?;
    let rows_updated = result.rows_affected();

    let delete_sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND chunk_id = $2"#,
        staging_schema,
    );
    sqlx::query(&delete_sql)
        .bind(epoch)
        .bind(chunk_id)
        .execute(&mut *tx)
        .await?;

    sqlx::query(
        "UPDATE rerand_progress SET live_applied = TRUE WHERE epoch = $1 AND chunk_id = $2",
    )
    .bind(epoch)
    .bind(chunk_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(rows_updated)
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
    let row: Option<(i32,)> = sqlx::query_as(
        "SELECT MAX(chunk_id) FROM rerand_progress WHERE epoch = $1 AND all_confirmed = TRUE",
    )
    .bind(epoch)
    .fetch_optional(pool)
    .await?;
    match row {
        Some((max,)) => Ok(Some(max)),
        None => Ok(None),
    }
}

/// Returns the highest epoch that has any rerand_progress rows.
pub async fn get_current_epoch(pool: &PgPool) -> Result<Option<i32>> {
    let row: (Option<i32>,) =
        sqlx::query_as("SELECT MAX(epoch) FROM rerand_progress")
            .fetch_one(pool)
            .await?;
    Ok(row.0)
}

/// Returns chunk_ids for a given epoch where live_applied = FALSE and
/// chunk_id <= up_to_chunk, ordered ascending.
pub async fn get_unapplied_chunks(
    pool: &PgPool,
    epoch: i32,
    up_to_chunk: i32,
) -> Result<Vec<i32>> {
    let rows: Vec<(i32,)> = sqlx::query_as(
        r#"
        SELECT chunk_id FROM rerand_progress
        WHERE epoch = $1 AND chunk_id <= $2 AND live_applied = FALSE
        ORDER BY chunk_id ASC
        "#,
    )
    .bind(epoch)
    .bind(up_to_chunk)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(id,)| id).collect())
}

// ---------------------------------------------------------------------------
// Shared startup helpers (used by both HNSW and GPU servers)
// ---------------------------------------------------------------------------

/// Build the rerand sync state from the local `rerand_progress` table.
pub async fn build_rerand_sync_state(pool: &PgPool) -> Result<RerandSyncState> {
    let epoch = get_current_epoch(pool).await?.unwrap_or(0);
    let max_confirmed = get_max_confirmed_chunk(pool, epoch)
        .await?
        .unwrap_or(-1);
    Ok(RerandSyncState {
        epoch,
        max_confirmed_chunk: max_confirmed,
    })
}

/// Compute the safe-to-apply watermark from all parties' rerand sync states.
/// Returns `Some((epoch, max_chunk_id))` if there are chunks to catch up,
/// `None` otherwise.
pub fn compute_rerand_safe_up_to(sync_result: &SyncResult) -> Result<Option<(i32, i32)>> {
    let my_state = match sync_result.my_state.rerand_state.as_ref() {
        Some(s) => s,
        None => return Ok(None),
    };
    let my_epoch = my_state.epoch;

    let rerand_states: Vec<&RerandSyncState> = sync_result
        .all_states
        .iter()
        .filter_map(|s| s.rerand_state.as_ref())
        .collect();

    if rerand_states.is_empty() {
        return Ok(None);
    }

    let mut safe_up_to = -1;
    for s in rerand_states {
        let diff = s.epoch - my_epoch;
        match diff {
            0 => {
                safe_up_to = safe_up_to.max(s.max_confirmed_chunk);
            }
            1 => {
                safe_up_to = i32::MAX;
            }
            -1 => {
                // They are behind, they contribute -1
            }
            _ => {
                eyre::bail!("Fatal epoch desync: local epoch is {}, but peer is on epoch {}", my_epoch, s.epoch);
            }
        }
    }

    if safe_up_to < 0 {
        return Ok(None);
    }

    Ok(Some((my_epoch, safe_up_to)))
}

/// Perform rerand catch-up and acquire the advisory lock.
///
/// 1. Computes the safe-to-apply watermark from `sync_result`.
/// 2. If there are unapplied chunks, acquires `pg_advisory_lock(RERAND_APPLY_LOCK)`
///    on a dedicated connection, then applies all unapplied chunks.
/// 3. Returns the lock-holding connection (if the lock was acquired).
///
/// The caller **must** keep the returned connection alive until `load_iris_db`
/// finishes, then call [`release_rerand_lock`] to release it.
pub async fn rerand_catchup_and_lock(
    pool: &PgPool,
    schema_name: &str,
    sync_result: &SyncResult,
) -> Result<Option<PoolConnection<Postgres>>> {
    let safe_up_to = match compute_rerand_safe_up_to(sync_result)? {
        Some(v) => v,
        None => return Ok(None),
    };

    let staging_schema = staging_schema_name(schema_name);
    tracing::info!(
        "Rerand catch-up: applying chunks up to {} for epoch {}",
        safe_up_to.1,
        safe_up_to.0
    );

    let mut conn = pool.acquire().await?;
    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(RERAND_APPLY_LOCK)
        .execute(&mut *conn)
        .await?;

    let unapplied = get_unapplied_chunks(pool, safe_up_to.0, safe_up_to.1).await?;
    for chunk_id in unapplied {
        let rows =
            apply_staging_chunk(pool, &staging_schema, safe_up_to.0, chunk_id).await?;
        tracing::info!(
            "Rerand catch-up: applied epoch {} chunk {} ({} rows)",
            safe_up_to.0,
            chunk_id,
            rows
        );
    }

    Ok(Some(conn))
}

/// Release the advisory lock acquired by [`rerand_catchup_and_lock`].
pub async fn release_rerand_lock(
    lock_conn: Option<PoolConnection<Postgres>>,
) -> Result<()> {
    if let Some(mut conn) = lock_conn {
        sqlx::query("SELECT pg_advisory_unlock($1)")
            .bind(RERAND_APPLY_LOCK)
            .execute(&mut *conn)
            .await?;
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
    fn test_compute_rerand_safe_up_to_same_epoch() {
        let p0 = dummy_sync_state(1, 5);
        let p1 = dummy_sync_state(1, 4);
        let p2 = dummy_sync_state(1, 6);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(compute_rerand_safe_up_to(&sync_result).unwrap(), Some((1, 6)));
    }

    #[test]
    fn test_compute_rerand_safe_up_to_peer_ahead() {
        // I am on epoch 0, but peer is on epoch 1.
        // This implies the peer has confirmed all my chunks for epoch 0.
        let p0 = dummy_sync_state(0, 5);
        let p1 = dummy_sync_state(1, 0); // ahead
        let p2 = dummy_sync_state(0, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(compute_rerand_safe_up_to(&sync_result).unwrap(), Some((0, i32::MAX)));
    }

    #[test]
    fn test_compute_rerand_safe_up_to_peer_behind() {
        // I am on epoch 1, but peer is on epoch 0.
        // This implies the peer has not confirmed any chunks for epoch 1.
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(0, 10); // behind
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(compute_rerand_safe_up_to(&sync_result).unwrap(), Some((1, 2)));
    }
    
    #[test]
    fn test_compute_rerand_safe_up_to_fatal_desync() {
        // I am on epoch 1, but peer is on epoch 3 (difference > 1).
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(3, 10); // way ahead
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(compute_rerand_safe_up_to(&sync_result).is_err());
    }
}
