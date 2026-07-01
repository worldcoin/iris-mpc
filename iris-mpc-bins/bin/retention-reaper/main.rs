//! retention-reaper — a generic, config-driven retention CronJob for append-only Postgres
//! tables. For each configured job it runs a bounded, batched, guarded `DELETE` of rows
//! older than a retention window, and emits Datadog/StatsD metrics (rows deleted, oldest
//! retained age, dead-tuple ratio, last success). One binary, N jobs via `RETENTION_JOBS`
//! config — reusable across anon_stats (POP-3905), modifications (POP-3931), and future
//! append-only stores. No table-specific code, no partition/catalog manipulation.
//!
//! Fully self-contained config (`RETENTION_*` env — see `ReaperConfig`). Depends only on
//! generic building blocks: `iris_mpc_common::postgres` for the pool (per-connection
//! `search_path`) and `ampc_server_utils::config::ServiceConfig` for the standard
//! Datadog/StatsD wiring. The DB it points at is deploy config, so the same image serves
//! any party / cluster / table-set.

use std::env;
use std::time::{Duration, Instant};

use ampc_server_utils::config::ServiceConfig;
use eyre::{bail, Context, Result};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::tracing::initialize_tracing;
use serde::Deserialize;
use sqlx::{Executor, PgConnection, PgPool, Row};
use tokio::time::timeout;
use tracing::{error, info, warn};

const DEFAULT_BATCH_SIZE: i64 = 10_000;

/// One retention job. `guard` is a raw SQL predicate ANDed into the DELETE — it is TRUSTED
/// deploy config (never user input); `table`/`ts_column` are validated as identifiers.
#[derive(Debug, Clone, Deserialize)]
struct RetentionJob {
    table: String,
    #[serde(default = "default_ts_column")]
    ts_column: String,
    /// Postgres interval literal, e.g. "14 days".
    retention: String,
    /// Optional extra predicate, e.g. "processed = TRUE" or
    /// "status = 'COMPLETED' AND persisted AND id < (SELECT ...)".
    #[serde(default)]
    guard: Option<String>,
    #[serde(default = "default_batch_size")]
    batch_size: i64,
}

fn default_ts_column() -> String {
    "created_at".to_string()
}
fn default_batch_size() -> i64 {
    DEFAULT_BATCH_SIZE
}

/// Self-contained config, deserialized from `RETENTION__*` env via the `config` crate —
/// the SAME mechanism the other services use for their `SMPC__*` config
/// (`Environment::with_prefix(...).separator("__").try_parsing(true)` → `try_deserialize`),
/// just its own prefix + struct. No dependency on any service-specific config, so the
/// reaper is portable across stores.
///
/// Env (all `RETENTION__`-prefixed, `__` = nesting, matching the `SMPC__…` convention):
///   `RETENTION__ENABLED` (default false — master kill switch),
///   `RETENTION__DB_URL` (required), `RETENTION__DB_SCHEMA` (default `public`),
///   `RETENTION__PARTY` (metrics label), `RETENTION__STATEMENT_TIMEOUT_SECS` (default 60),
///   `RETENTION__SERVICE__SERVICE_NAME` + `RETENTION__SERVICE__METRICS__{HOST,PORT,
///   QUEUE_SIZE,BUFFER_SIZE,PREFIX}` (optional; enables Datadog/StatsD, same shape as the
///   servers' `SMPC__SERVICE__METRICS__…`).
/// `RETENTION_JOBS` (single `_`, a JSON array) is loaded separately — a `Vec` of structs
/// isn't expressible as flat env, and the single `_` keeps it outside the `__` scheme so
/// the `config` Environment source ignores it.
#[derive(Debug, Deserialize)]
struct ReaperConfig {
    /// `RETENTION__ENABLED` (default `false`) — master kill switch. When false the job
    /// logs and exits cleanly without connecting to the DB or touching any table.
    #[serde(default)]
    enabled: bool,
    db_url: String,
    #[serde(default = "default_schema")]
    db_schema: String,
    #[serde(default)]
    party: String,
    #[serde(default = "default_statement_timeout_secs")]
    statement_timeout_secs: u64,
    #[serde(default)]
    service: Option<ServiceConfig>,
    #[serde(default, skip)]
    jobs: Vec<RetentionJob>,
}

fn default_schema() -> String {
    "public".to_string()
}
fn default_statement_timeout_secs() -> u64 {
    60
}

impl ReaperConfig {
    fn load() -> Result<Self> {
        let mut config: ReaperConfig = config::Config::builder()
            .add_source(
                config::Environment::with_prefix("RETENTION")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()
            .wrap_err("failed to build RETENTION__* config")?
            .try_deserialize()
            .wrap_err(
                "failed to deserialize RETENTION__* env config (RETENTION__DB_URL required)",
            )?;
        config.jobs = load_jobs()?;
        Ok(config)
    }

    fn statement_timeout(&self) -> Duration {
        Duration::from_secs(self.statement_timeout_secs)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let config = ReaperConfig::load()?;
    let _tracing =
        initialize_tracing(config.service.clone()).wrap_err("failed to initialize tracing")?;

    // Master kill switch — disabled by default. Log and exit cleanly (exit 0) without
    // connecting to the DB, so enabling retention on a cluster is a deliberate flag flip.
    if !config.enabled {
        info!(
            party = %config.party,
            "retention-reaper is disabled (RETENTION__ENABLED=false) — exiting without action"
        );
        return Ok(());
    }

    let party = config.party.clone();
    let statement_timeout = config.statement_timeout();

    let client = PostgresClient::new(&config.db_url, &config.db_schema, AccessMode::ReadWrite)
        .await
        .wrap_err("failed to connect to postgres")?;

    info!(
        party = %party,
        schema = %config.db_schema,
        jobs = config.jobs.len(),
        "retention-reaper starting"
    );

    let start = Instant::now();
    let mut failed = false;
    for job in &config.jobs {
        // table + ts_column are interpolated into SQL, so must be strict identifiers.
        validate_identifier(&job.table)
            .and_then(|_| validate_identifier(&job.ts_column))
            .wrap_err_with(|| format!("invalid identifier in job for table {}", job.table))?;
        match reap(&client.pool, job, statement_timeout, &party).await {
            Ok(deleted) => info!(table = %job.table, rows_deleted = deleted, "retention job ok"),
            Err(error) => {
                error!(table = %job.table, error = %error, "retention job failed");
                failed = true; // keep going: one bad table shouldn't skip the others
            }
        }
    }

    metrics::histogram!("retention.run.duration", "party" => party.clone())
        .record(start.elapsed().as_secs_f64());

    if failed {
        // non-zero exit → CronJob marks the run failed → missing-run/failure alert fires.
        bail!("one or more retention jobs failed");
    }
    metrics::gauge!("retention.last_success", "party" => party.clone()).set(1.0);
    info!(
        elapsed_seconds = start.elapsed().as_secs_f64(),
        "retention-reaper completed"
    );
    Ok(())
}

/// Batched, guarded delete of rows older than the retention window. Deletes in `batch_size`
/// chunks by `ctid` so each statement is short (bounded lock, autovacuum-friendly) and
/// loops until a partial batch signals the table is drained for this run.
async fn reap(
    pool: &PgPool,
    job: &RetentionJob,
    statement_timeout: Duration,
    party: &str,
) -> Result<i64> {
    let guard = job
        .guard
        .as_deref()
        .map(|g| format!(" AND ({g})"))
        .unwrap_or_default();
    // retention is bound as a parameter; identifiers are validated + quoted; batch is i64.
    let delete_sql = format!(
        "WITH del AS (DELETE FROM {table} WHERE ctid IN (\
             SELECT ctid FROM {table} WHERE {ts} < now() - $1::interval{guard} LIMIT {batch}\
         ) RETURNING 1) SELECT count(*)::bigint AS n FROM del",
        table = quote_ident(&job.table),
        ts = quote_ident(&job.ts_column),
        guard = guard,
        batch = job.batch_size,
    );

    let mut conn = pool.acquire().await.wrap_err("acquire connection")?;
    set_statement_timeout(&mut conn, statement_timeout).await?;

    let mut total: i64 = 0;
    loop {
        let deleted: i64 = timeout(
            statement_timeout + Duration::from_secs(5),
            sqlx::query(&delete_sql)
                .bind(&job.retention)
                .fetch_one(&mut *conn),
        )
        .await
        .wrap_err_with(|| format!("delete batch on {} exceeded timeout", job.table))?
        .wrap_err_with(|| format!("delete batch on {} failed", job.table))?
        .try_get::<i64, _>("n")?;

        total += deleted;
        if deleted < job.batch_size {
            break;
        }
    }

    metrics::counter!(
        "retention.rows_deleted",
        "table" => job.table.clone(),
        "party" => party.to_string()
    )
    .increment(total as u64);
    // Health gauges are best-effort: a probe failure must never fail the run or block deletes.
    emit_health_metrics(pool, job, party).await;
    Ok(total)
}

/// Oldest-retained age (retention actually working) + dead-tuple ratio (bloat — the one real
/// risk of DELETE-based retention). Both best-effort; logged and skipped on error.
async fn emit_health_metrics(pool: &PgPool, job: &RetentionJob, party: &str) {
    // Oldest row *among the retention-eligible rows* — apply the same guard as the delete.
    // Without it, an intentionally-retained old row (e.g. a still-unprocessed anon_stats row
    // under guard `processed = TRUE`) would inflate this gauge and falsely trip the
    // retention-lag monitor even though the reaper is working correctly.
    let guard_where = job
        .guard
        .as_deref()
        .map(|g| format!(" WHERE ({g})"))
        .unwrap_or_default();
    let oldest_sql = format!(
        "SELECT EXTRACT(EPOCH FROM (now() - MIN({ts})))::float8 AS age FROM {table}{guard_where}",
        ts = quote_ident(&job.ts_column),
        table = quote_ident(&job.table),
        guard_where = guard_where,
    );
    match sqlx::query(&oldest_sql).fetch_one(pool).await {
        Ok(row) => {
            if let Ok(age) = row.try_get::<Option<f64>, _>("age") {
                metrics::gauge!(
                    "retention.oldest_retained_seconds",
                    "table" => job.table.clone(), "party" => party.to_string()
                )
                .set(age.unwrap_or(0.0));
            }
        }
        Err(e) => warn!(table = %job.table, error = %e, "oldest_retained probe failed"),
    }

    let bloat_sql = "SELECT n_dead_tup, n_live_tup FROM pg_stat_user_tables WHERE relname = $1";
    match sqlx::query(bloat_sql)
        .bind(&job.table)
        .fetch_optional(pool)
        .await
    {
        Ok(Some(row)) => {
            let dead: i64 = row.try_get("n_dead_tup").unwrap_or(0);
            let live: i64 = row.try_get("n_live_tup").unwrap_or(0);
            let ratio = if dead + live > 0 {
                dead as f64 / (dead + live) as f64
            } else {
                0.0
            };
            metrics::gauge!(
                "retention.dead_tuple_ratio",
                "table" => job.table.clone(), "party" => party.to_string()
            )
            .set(ratio);
        }
        Ok(None) => {}
        Err(e) => warn!(table = %job.table, error = %e, "dead_tuple probe failed"),
    }
}

async fn set_statement_timeout(conn: &mut PgConnection, timeout: Duration) -> Result<()> {
    conn.execute(format!("SET statement_timeout = {}", timeout.as_millis()).as_str())
        .await
        .wrap_err("failed to set statement_timeout")?;
    Ok(())
}

fn load_jobs() -> Result<Vec<RetentionJob>> {
    let raw = env::var("RETENTION_JOBS")
        .wrap_err("RETENTION_JOBS env var is required (JSON array of retention jobs)")?;
    let jobs: Vec<RetentionJob> =
        serde_json::from_str(&raw).wrap_err("RETENTION_JOBS is not valid JSON")?;
    if jobs.is_empty() {
        bail!("RETENTION_JOBS is empty — nothing to do");
    }
    Ok(jobs)
}

/// Postgres identifier guard: unqualified, `[A-Za-z_][A-Za-z0-9_]*`. Rejects anything that
/// could break out of the interpolated position (dots, quotes, whitespace, etc.).
fn validate_identifier(ident: &str) -> Result<()> {
    let ok = !ident.is_empty()
        && ident.len() <= 63
        && ident
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        Ok(())
    } else {
        bail!("invalid SQL identifier: {ident:?}")
    }
}

fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_plain_identifiers() {
        assert!(validate_identifier("anon_stats_1d").is_ok());
        assert!(validate_identifier("created_at").is_ok());
    }

    #[test]
    fn rejects_injection_identifiers() {
        for bad in ["anon; DROP TABLE x", "a.b", "\"quoted\"", "", "1d", "a b"] {
            assert!(validate_identifier(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn parses_jobs_with_defaults_and_guard() {
        let raw = r#"[
          {"table":"anon_stats_1d","retention":"14 days","guard":"processed = TRUE"},
          {"table":"modifications","ts_column":"created_at","retention":"30 days",
           "guard":"status = 'COMPLETED' AND persisted","batch_size":5000}
        ]"#;
        let jobs: Vec<RetentionJob> = serde_json::from_str(raw).unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].ts_column, "created_at"); // defaulted
        assert_eq!(jobs[0].batch_size, DEFAULT_BATCH_SIZE); // defaulted
        assert_eq!(jobs[0].guard.as_deref(), Some("processed = TRUE"));
        assert_eq!(jobs[1].batch_size, 5000);
    }

    #[test]
    fn quotes_identifiers() {
        assert_eq!(quote_ident("anon_stats_1d"), "\"anon_stats_1d\"");
    }
}
