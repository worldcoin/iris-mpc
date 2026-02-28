use std::time::Duration;

use eyre::Result;
use iris_mpc_common::helpers::sync::RerandSyncState;
use sqlx::PgPool;

pub const RERAND_APPLY_LOCK: i64 = 0x5245_5241_4E44;
pub const RERAND_MODIFY_LOCK: i64 = 0x5245_4D4F_4446;

/// Acquire `RERAND_MODIFY_LOCK` as a transaction-level advisory lock.
/// Auto-released on commit/rollback.
pub async fn acquire_modify_lock(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> Result<()> {
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(RERAND_MODIFY_LOCK)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

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

async fn delete_staging_ids_tx(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
    ids: &[i64],
) -> Result<u64> {
    if ids.is_empty() {
        return Ok(0);
    }
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch = $1 AND chunk_id = $2 AND id = ANY($3)"#,
        staging_schema,
    );
    let result = sqlx::query(&sql)
        .bind(epoch)
        .bind(chunk_id)
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

/// Apply a confirmed staging chunk to the live `irises` table.
///
/// Opens a single transaction that:
/// 1. Acquires `RERAND_MODIFY_LOCK` (blocks modification writes)
/// 2. Acquires `RERAND_APPLY_LOCK` (blocks startup DB load)
/// 3. Deletes `staging_divergent` IDs from staging (cross-party disagreements)
/// 4. Applies remaining staging rows via `version_id` CAS
/// 5. Cleans up staging and marks progress
///
/// The `version_id` CAS (`WHERE irises.version_id = staging.original_version_id`)
/// silently skips any rows that were modified between staging and apply. This is
/// safe: the modification will propagate to all parties and overwrite whatever
/// was there, restoring consistency. See the spec's "Conflict Resolution" section.
pub async fn apply_confirmed_chunk(
    pool: &PgPool,
    staging_schema: &str,
    epoch: i32,
    chunk_id: i32,
    staging_divergent: &[i64],
) -> Result<u64> {
    validate_identifier(staging_schema)?;
    let mut tx = pool.begin().await?;

    acquire_modify_lock(&mut tx).await?;
    sqlx::query("SELECT pg_advisory_xact_lock($1)")
        .bind(RERAND_APPLY_LOCK)
        .execute(&mut *tx)
        .await?;

    if !staging_divergent.is_empty() {
        let deleted =
            delete_staging_ids_tx(&mut tx, staging_schema, epoch, chunk_id, staging_divergent)
                .await?;
        tracing::info!(
            "Rerand apply: removed {} staging-divergent rows (epoch={}, chunk={})",
            deleted,
            epoch,
            chunk_id,
        );
    }

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

/// Returns the highest `chunk_id` where `live_applied = TRUE` for a given
/// epoch, or `None` if no chunks have been applied in that epoch yet.
pub async fn get_max_applied_chunk_for_epoch(
    pool: &PgPool,
    epoch: i32,
) -> Result<Option<i32>> {
    let row: (Option<i32>,) = sqlx::query_as(
        "SELECT MAX(chunk_id) FROM rerand_progress WHERE epoch = $1 AND live_applied = TRUE",
    )
    .bind(epoch)
    .fetch_one(pool)
    .await?;
    Ok(row.0)
}

/// Delete all staging rows for epochs older than `current_epoch`.
pub async fn delete_staging_for_old_epochs(
    pool: &PgPool,
    staging_schema: &str,
    current_epoch: i32,
) -> Result<u64> {
    validate_identifier(staging_schema)?;
    let sql = format!(
        r#"DELETE FROM "{}".irises WHERE epoch < $1"#,
        staging_schema
    );
    let result = sqlx::query(&sql)
        .bind(current_epoch)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

/// Delete rerand progress rows for epochs older than `current_epoch`.
pub async fn delete_rerand_progress_for_old_epochs(pool: &PgPool, current_epoch: i32) -> Result<u64> {
    let result = sqlx::query("DELETE FROM rerand_progress WHERE epoch < $1")
        .bind(current_epoch)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
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
/// (rolling deploy before migration). Returns `Err` for real DB failures.
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
    let max_applied = get_max_applied_chunk_for_epoch(pool, epoch).await?.unwrap_or(-1);
    Ok(Some(RerandSyncState {
        epoch,
        max_applied_chunk: max_applied,
    }))
}

fn is_undefined_table(err: &eyre::Report) -> bool {
    if let Some(db_err) = err.root_cause().downcast_ref::<sqlx::Error>() {
        return is_undefined_table_sqlx(db_err);
    }
    false
}

fn is_undefined_table_sqlx(err: &sqlx::Error) -> bool {
    if let sqlx::Error::Database(pg) = err {
        return pg.code().as_deref() == Some("42P01");
    }
    false
}


// ---------------------------------------------------------------------------
// Freeze protocol: coordinated pause of the rerand worker during startup
// ---------------------------------------------------------------------------

const FREEZE_TIMEOUT: Duration = Duration::from_secs(120);
const FREEZE_POLL: Duration = Duration::from_secs(2);

fn rerand_control_exists(err: &sqlx::Error) -> bool {
    !is_undefined_table_sqlx(err)
}

/// Request the rerand worker to freeze. Writes a unique `freeze_generation`
/// to `rerand_control`. Returns the generation token.
pub async fn request_rerand_freeze(pool: &PgPool) -> Result<Option<String>> {
    let generation = uuid::Uuid::new_v4().to_string();
    match sqlx::query(
        "UPDATE rerand_control SET freeze_requested = TRUE, freeze_generation = $1, frozen_generation = NULL WHERE id = 1",
    )
    .bind(&generation)
    .execute(pool)
    .await
    {
        Ok(_) => Ok(Some(generation)),
        Err(e) if !rerand_control_exists(&e) => {
            tracing::info!("rerand_control table missing; skipping freeze (pre-migration)");
            Ok(None)
        }
        Err(e) => Err(e.into()),
    }
}

/// Wait until the rerand worker acknowledges the freeze by writing
/// `frozen_generation = generation`. Fails closed on timeout.
pub async fn wait_for_rerand_frozen(pool: &PgPool, generation: &str) -> Result<()> {
    let deadline = tokio::time::Instant::now() + FREEZE_TIMEOUT;
    loop {
        let row: Option<(Option<String>,)> = sqlx::query_as(
            "SELECT frozen_generation FROM rerand_control WHERE id = 1",
        )
        .fetch_optional(pool)
        .await?;

        if let Some((Some(frozen_gen),)) = row {
            if frozen_gen == generation {
                tracing::info!("Rerand worker confirmed freeze (generation={})", generation);
                return Ok(());
            }
        }

        if tokio::time::Instant::now() >= deadline {
            eyre::bail!(
                "Rerand worker did not acknowledge freeze after {:?} (generation={}). \
                 Ensure the rerand worker is running and healthy.",
                FREEZE_TIMEOUT,
                generation,
            );
        }
        tokio::time::sleep(FREEZE_POLL).await;
    }
}

/// Called by the rerand worker between chunks. If a freeze is requested,
/// acknowledge it and block until the freeze is lifted. Returns `true` if
/// the worker should continue, `false` if cancelled while frozen.
pub async fn check_and_handle_freeze(
    pool: &PgPool,
    cancel: Option<&tokio_util::sync::CancellationToken>,
) -> Result<bool> {
    let row: Option<(bool, Option<String>)> = match sqlx::query_as(
        "SELECT freeze_requested, freeze_generation FROM rerand_control WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    {
        Ok(r) => r,
        Err(e) if !rerand_control_exists(&e) => return Ok(true),
        Err(e) => return Err(e.into()),
    };

    let Some((true, Some(generation))) = row else {
        return Ok(true);
    };

    tracing::info!("Rerand freeze requested (generation={}), pausing...", generation);

    // Acknowledge the freeze.
    sqlx::query("UPDATE rerand_control SET frozen_generation = $1 WHERE id = 1")
        .bind(&generation)
        .execute(pool)
        .await?;

    let mut current_gen = generation.to_string();

    // Block until freeze is lifted. Re-read freeze_generation each iteration
    // so that if the requesting server crashes and restarts with a new
    // generation, we re-acknowledge instead of deadlocking.
    loop {
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Ok(false);
        }

        let row: Option<(bool, Option<String>)> = sqlx::query_as(
            "SELECT freeze_requested, freeze_generation FROM rerand_control WHERE id = 1",
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some((false, _)) | None => {
                tracing::info!("Rerand freeze lifted, resuming");
                return Ok(true);
            }
            Some((true, Some(ref new_gen))) if *new_gen != current_gen => {
                tracing::info!(
                    "Rerand freeze generation changed ({} -> {}), re-acknowledging",
                    current_gen,
                    new_gen
                );
                sqlx::query("UPDATE rerand_control SET frozen_generation = $1 WHERE id = 1")
                    .bind(new_gen)
                    .execute(pool)
                    .await?;
                current_gen = new_gen.clone();
            }
            _ => {}
        }

        tokio::time::sleep(FREEZE_POLL).await;
    }
}

/// Lift the freeze and clear the generation. Called after `load_iris_db`.
/// Retries on transient DB errors to avoid leaving the worker permanently frozen.
/// Silently succeeds if the `rerand_control` table doesn't exist (pre-migration).
pub async fn release_rerand_freeze(pool: &PgPool) -> Result<()> {
    for attempt in 0..5 {
        match sqlx::query(
            "UPDATE rerand_control SET freeze_requested = FALSE, freeze_generation = NULL, frozen_generation = NULL WHERE id = 1",
        )
        .execute(pool)
        .await
        {
            Ok(_) => {
                tracing::info!("Rerand freeze released");
                return Ok(());
            }
            Err(e) if !rerand_control_exists(&e) => {
                return Ok(());
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to release rerand freeze (attempt {}): {:?}",
                    attempt + 1,
                    e
                );
                tokio::time::sleep(FREEZE_POLL).await;
            }
        }
    }
    eyre::bail!("Failed to release rerand freeze after 5 attempts — worker may be stuck frozen");
}

/// Acquire `RERAND_APPLY_LOCK` on a detached connection. The lock is held
/// through `load_iris_db` to prevent any concurrent rerand applies (belt
/// and suspenders — the freeze should already have paused the worker).
pub async fn acquire_apply_lock(pool: &PgPool) -> Result<Option<sqlx::PgConnection>> {
    let mut conn = pool.acquire().await?;

    // If rerand tables don't exist yet, skip.
    match sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM rerand_progress LIMIT 1",
    )
    .fetch_one(&mut *conn)
    .await
    {
        Err(e) if is_undefined_table_sqlx(&e) => return Ok(None),
        Err(e) => return Err(e.into()),
        Ok(_) => {}
    }

    sqlx::query("SELECT pg_advisory_lock($1)")
        .bind(RERAND_APPLY_LOCK)
        .execute(&mut *conn)
        .await?;

    Ok(Some(conn.detach()))
}

/// Release the advisory lock and close the connection.
pub async fn release_apply_lock(lock_conn: Option<sqlx::PgConnection>) -> Result<()> {
    if let Some(mut conn) = lock_conn {
        let _ = sqlx::query("SELECT pg_advisory_unlock($1)")
            .bind(RERAND_APPLY_LOCK)
            .execute(&mut conn)
            .await;
        drop(conn);
        tracing::info!("RERAND_APPLY_LOCK released after DB load");
    }
    Ok(())
}

/// Get the local applied watermark: `(epoch, max_chunk_id)` where
/// `live_applied = TRUE`. Returns `None` pre-migration or if no chunks
/// have been applied.
pub async fn get_applied_watermark_from_pool(pool: &PgPool) -> Result<Option<(i32, i32)>> {
    let row: Option<(i32, i32)> = match sqlx::query_as(
        "SELECT epoch, chunk_id FROM rerand_progress \
         WHERE live_applied = TRUE \
         ORDER BY epoch DESC, chunk_id DESC \
         LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    {
        Ok(row) => row,
        Err(e) if is_undefined_table_sqlx(&e) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    Ok(row)
}

async fn fetch_peer_watermark(host: &str, port: usize) -> Result<Option<(i32, i32)>> {
    let url = format!("http://{}:{}/rerand-watermark", host, port);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| eyre::eyre!("Failed to reach {} for watermark: {}", url, e))?;
    if !resp.status().is_success() {
        eyre::bail!(
            "Peer {} returned HTTP {} for watermark",
            url,
            resp.status()
        );
    }
    let body = resp
        .text()
        .await
        .map_err(|e| eyre::eyre!("Failed to read watermark from {}: {}", url, e))?;

    if body.trim() == "null" {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| eyre::eyre!("Failed to parse watermark from {}: {}", url, e))?;
    Ok(Some((
        v["epoch"]
            .as_i64()
            .ok_or_else(|| eyre::eyre!("Missing epoch in watermark from {}", url))?
            as i32,
        v["max_applied_chunk"]
            .as_i64()
            .ok_or_else(|| eyre::eyre!("Missing max_applied_chunk in watermark from {}", url))?
            as i32,
    )))
}

/// Freeze the local rerand worker, then verify all peers report the exact
/// same applied watermark. If this party is behind, release the freeze
/// briefly so the worker can catch up, then re-freeze and re-check.
/// If this party is at the max, stay frozen and wait for peers to catch up.
///
/// Guarantees: when this returns `Ok(())`, the local worker is frozen and
/// all parties have the same `(epoch, max_applied_chunk)`.
/// On any error, the freeze is released before the error propagates.
pub async fn freeze_and_verify_watermarks(
    pool: &PgPool,
    peers: &[(&str, usize)],
) -> Result<()> {
    if peers.is_empty() {
        eyre::bail!("freeze_and_verify_watermarks called with no peers");
    }

    let result = freeze_and_verify_inner(pool, peers).await;
    if result.is_err() {
        if let Err(release_err) = release_rerand_freeze(pool).await {
            tracing::error!(
                "Failed to release rerand freeze during error cleanup: {:?}. \
                 Worker may be stuck frozen until next successful startup.",
                release_err
            );
        }
    }
    result
}

async fn freeze_and_verify_inner(
    pool: &PgPool,
    peers: &[(&str, usize)],
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + FREEZE_TIMEOUT;

    loop {
        let gen = match request_rerand_freeze(pool).await? {
            Some(g) => g,
            None => return Ok(()), // pre-migration, no rerand tables
        };
        wait_for_rerand_frozen(pool, &gen).await?;

        loop {
            if tokio::time::Instant::now() >= deadline {
                release_rerand_freeze(pool).await?;
                eyre::bail!(
                    "Rerand watermark convergence timeout after {:?}. \
                     Ensure all rerand workers and main servers are healthy.",
                    FREEZE_TIMEOUT,
                );
            }

            let local = get_applied_watermark_from_pool(pool).await?;
            let mut all_equal = true;
            let mut max_wm = local;

            for (host, port) in peers {
                let peer = fetch_peer_watermark(host, *port).await?;
                if peer != local {
                    all_equal = false;
                }
                if peer > max_wm {
                    max_wm = peer;
                }
            }

            if all_equal {
                tracing::info!(
                    "Rerand watermark equality confirmed across all parties: {:?}",
                    local
                );
                return Ok(());
            }

            if local < max_wm {
                tracing::info!(
                    "Local watermark {:?} behind max {:?}, releasing freeze to catch up",
                    local,
                    max_wm
                );
                release_rerand_freeze(pool).await?;
                tokio::time::sleep(Duration::from_secs(5)).await;
                break; // outer loop will re-freeze and re-check
            }

            tracing::info!(
                "Local watermark {:?} at max, waiting for peers to catch up...",
                local
            );
            tokio::time::sleep(FREEZE_POLL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_staging_schema_name() {
        assert_eq!(staging_schema_name("public"), "public_rerand_staging");
    }

    #[test]
    fn test_validate_identifier_ok() {
        assert!(validate_identifier("public_rerand_staging").is_ok());
    }

    #[test]
    fn test_validate_identifier_rejects_injection() {
        assert!(validate_identifier("public; DROP TABLE irises").is_err());
    }
}
