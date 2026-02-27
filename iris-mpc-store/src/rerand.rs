use eyre::Result;
use iris_mpc_common::helpers::sync::{RerandSyncState, SyncResult};
use sqlx::PgPool;

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

pub async fn ensure_staging_schema(pool: &PgPool, staging_schema: &str) -> Result<()> {
    validate_identifier(staging_schema)?;
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
///   1. UPDATE live irises from staging (optimistic lock on version_id)
///   2. DELETE staging rows for this chunk
///   3. Mark live_applied in rerand_progress
pub async fn apply_staging_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
) -> Result<u64> {
    validate_identifier(staging_schema)?;
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
pub async fn build_rerand_sync_state(pool: &PgPool) -> Result<RerandSyncState> {
    let epoch = get_current_epoch(pool).await?.unwrap_or(0);
    let max_confirmed = get_max_confirmed_chunk(pool, epoch).await?.unwrap_or(-1);
    Ok(RerandSyncState {
        epoch,
        max_confirmed_chunk: max_confirmed,
    })
}

/// Compute the single chunk (if any) that needs to be applied during startup catch-up.
///
/// Because the rerand loop has a strict per-chunk synchronization barrier (all 3 parties
/// must confirm chunk K before any party can stage chunk K+1), peers can be at most
/// 1 confirmed chunk ahead. Therefore, catch-up is always 0 or 1 chunks.
///
/// Returns `Some((epoch, chunk_id))` if there is exactly one chunk to catch up,
/// `None` otherwise.
pub fn compute_rerand_catchup_chunk(sync_result: &SyncResult) -> Result<Option<(i32, i32)>> {
    let my_state = match sync_result.my_state.rerand_state.as_ref() {
        Some(s) => s,
        None => return Ok(None),
    };
    let my_epoch = my_state.epoch;
    let my_chunk = my_state.max_confirmed_chunk;

    let mut any_peer_ahead = false;

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
                if chunk_diff == 1 {
                    any_peer_ahead = true;
                }
            }
            1 => {
                any_peer_ahead = true;
            }
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

    if !any_peer_ahead {
        return Ok(None);
    }

    let catchup_chunk = my_chunk + 1;
    Ok(Some((my_epoch, catchup_chunk)))
}

/// Perform rerand catch-up and acquire the advisory lock.
///
/// 1. Determines whether this node is 1 chunk behind a peer.
/// 2. If so, acquires `pg_advisory_lock(RERAND_APPLY_LOCK)` on a dedicated
///    connection and applies the single missing chunk.
/// 3. Returns the lock-holding connection (caller keeps it alive through
///    `load_iris_db`, then calls [`release_rerand_lock`]).
pub async fn rerand_catchup_and_lock(
    pool: &PgPool,
    schema_name: &str,
    sync_result: &SyncResult,
) -> Result<Option<sqlx::pool::PoolConnection<sqlx::Postgres>>> {
    let mut conn = pool.acquire().await?;
    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(RERAND_APPLY_LOCK)
        .execute(&mut *conn)
        .await?;

    if let Some((epoch, chunk_id)) = compute_rerand_catchup_chunk(sync_result)? {
        let staging_schema = staging_schema_name(schema_name);
        tracing::info!(
            "Rerand catch-up: applying epoch {} chunk {}",
            epoch,
            chunk_id,
        );

        let rows = match apply_staging_chunk(pool, &staging_schema, epoch, chunk_id).await {
            Ok(r) => r,
            Err(e) => {
                let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
                    .bind(RERAND_APPLY_LOCK)
                    .execute(&mut *conn)
                    .await;
                return Err(e);
            }
        };
        tracing::info!(
            "Rerand catch-up: applied epoch {} chunk {} ({} rows)",
            epoch,
            chunk_id,
            rows
        );
    } else {
        tracing::info!("Rerand catch-up: no chunk to apply");
    }

    Ok(Some(conn))
}

/// Release the advisory lock acquired by [`rerand_catchup_and_lock`].
pub async fn release_rerand_lock(
    lock_conn: Option<sqlx::pool::PoolConnection<sqlx::Postgres>>,
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
    fn test_catchup_peer_one_chunk_ahead() {
        let p0 = dummy_sync_state(1, 4);
        let p1 = dummy_sync_state(1, 4);
        let p2 = dummy_sync_state(1, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(
            compute_rerand_catchup_chunk(&sync_result).unwrap(),
            Some((1, 5))
        );
    }

    #[test]
    fn test_catchup_all_same() {
        let p0 = dummy_sync_state(1, 5);
        let p1 = dummy_sync_state(1, 5);
        let p2 = dummy_sync_state(1, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(compute_rerand_catchup_chunk(&sync_result).unwrap(), None);
    }

    #[test]
    fn test_catchup_peer_epoch_ahead() {
        let p0 = dummy_sync_state(0, 5);
        let p1 = dummy_sync_state(1, 0);
        let p2 = dummy_sync_state(0, 5);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(
            compute_rerand_catchup_chunk(&sync_result).unwrap(),
            Some((0, 6))
        );
    }

    #[test]
    fn test_catchup_peer_epoch_behind() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(0, 10);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert_eq!(compute_rerand_catchup_chunk(&sync_result).unwrap(), None);
    }

    #[test]
    fn test_catchup_fatal_chunk_desync() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(1, 4);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(compute_rerand_catchup_chunk(&sync_result).is_err());
    }

    #[test]
    fn test_catchup_fatal_epoch_desync() {
        let p0 = dummy_sync_state(1, 2);
        let p1 = dummy_sync_state(3, 10);
        let p2 = dummy_sync_state(1, 2);
        let sync_result = SyncResult {
            my_state: p0.clone(),
            all_states: vec![p0, p1, p2],
        };
        assert!(compute_rerand_catchup_chunk(&sync_result).is_err());
    }
}
