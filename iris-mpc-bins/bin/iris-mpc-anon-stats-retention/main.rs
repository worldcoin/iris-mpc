use std::env;
use std::time::Duration;

use ampc_anon_stats::store::postgres::AccessMode as AnonStatsAccessMode;
use ampc_anon_stats::store::postgres::PostgresClient as AnonStatsPgClient;
use ampc_anon_stats::AnonStatsServerConfig;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use eyre::{eyre, Context, Result};
use iris_mpc_common::tracing::initialize_tracing;
use sqlx::postgres::PgRow;
use sqlx::{Executor, PgConnection, PgPool, Row};
use tokio::time::timeout;
use tracing::{error, info, warn};

const DEFAULT_RETENTION_DAYS: i64 = 14;
const DEFAULT_PREMAKE_DAYS: i64 = 7;
const DEFAULT_QUARANTINE_GRACE_HOURS: i64 = 48;
const DEFAULT_STATEMENT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_DETACH_TIMEOUT_SECS: u64 = 300;
const DEFAULT_TABLES: [&str; 5] = [
    "anon_stats_1d",
    "anon_stats_1d_lifted",
    "anon_stats_2d",
    "anon_stats_2d_lifted",
    "anon_stats_face",
];

#[derive(Clone, Debug)]
struct RetentionConfig {
    tables: Vec<String>,
    retention_days: i64,
    premake_days: i64,
    quarantine_grace_hours: i64,
    statement_timeout: Duration,
    detach_timeout: Duration,
}

#[derive(Debug)]
struct RangePartition {
    name: String,
    lower_bound: DateTime<Utc>,
    upper_bound: DateTime<Utc>,
}

#[derive(Debug)]
struct QuarantineRelation {
    name: String,
    detach_date: NaiveDate,
}

#[derive(Debug)]
struct RunContext {
    party: String,
    today: NaiveDate,
    now: DateTime<Utc>,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let config = AnonStatsServerConfig::load_config("SMPC")
        .wrap_err("failed to load SMPC anon-stats server config")?;
    let retention_config = RetentionConfig::from_env()?;
    let _tracing_shutdown_handle =
        initialize_tracing(config.service.clone()).wrap_err("Failed to initialize tracing")?;

    let postgres_client = AnonStatsPgClient::new(
        &config.db_url,
        &config.db_schema_name,
        AnonStatsAccessMode::ReadWrite,
    )
    .await
    .wrap_err("failed to create anon-stats postgres client")?;

    let start = std::time::Instant::now();
    let run_context = RunContext {
        party: config.party_id.to_string(),
        today: Utc::now().date_naive(),
        now: Utc::now(),
    };

    match run_retention(
        &postgres_client.pool,
        &retention_config,
        &run_context,
        &config.db_schema_name,
    )
    .await
    {
        Ok(()) => {
            metrics::histogram!(
                "anon_stats_retention.run.duration",
                "party" => run_context.party.clone()
            )
            .record(start.elapsed().as_secs_f64());
            metrics::gauge!(
                "anon_stats_retention.last_success",
                "party" => run_context.party.clone()
            )
            .set(1.0);
            info!(
                operation = "retention_run",
                party_id = config.party_id,
                elapsed_seconds = start.elapsed().as_secs_f64(),
                "anon-stats retention run completed"
            );
            Ok(())
        }
        Err(error) => {
            metrics::histogram!(
                "anon_stats_retention.run.duration",
                "party" => run_context.party.clone()
            )
            .record(start.elapsed().as_secs_f64());
            error!(
                operation = "retention_run",
                party_id = config.party_id,
                elapsed_seconds = start.elapsed().as_secs_f64(),
                error = %error,
                "anon-stats retention run failed"
            );
            Err(error)
        }
    }
}

async fn run_retention(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    schema_name: &str,
) -> Result<()> {
    for table in &config.tables {
        validate_identifier(table).with_context(|| format!("invalid table identifier: {table}"))?;
        finalize_pending_detach_partitions(pool, config, table, &context.party)
            .await
            .with_context(|| {
                format!(
                    "table={table} operation=finalize_pending_detaches party={}",
                    context.party
                )
            })?;
    }

    for table in &config.tables {
        validate_identifier(table).with_context(|| format!("invalid table identifier: {table}"))?;
        info!(
            table = table.as_str(),
            party_id = context.party.as_str(),
            operation = "table_retention_start",
            "starting table retention"
        );

        premake_partitions(pool, config, context, table)
            .await
            .with_context(|| format!("table={table} operation=premake party={}", context.party))?;
        reap_range_partitions(pool, config, context, table)
            .await
            .with_context(|| {
                format!(
                    "table={table} operation=reap_range_partitions party={}",
                    context.party
                )
            })?;
        drop_expired_quarantine_relations(pool, config, context, schema_name, table)
            .await
            .with_context(|| {
                format!(
                    "table={table} operation=drop_expired_quarantine party={}",
                    context.party
                )
            })?;
        reap_legacy_default_partition(pool, config, context, table)
            .await
            .with_context(|| {
                format!(
                    "table={table} operation=reap_legacy_default party={}",
                    context.party
                )
            })?;
        emit_live_partition_metrics(pool, config, context, table)
            .await
            .with_context(|| {
                format!(
                    "table={table} operation=emit_live_partition_metrics party={}",
                    context.party
                )
            })?;
    }

    Ok(())
}

async fn premake_partitions(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    table: &str,
) -> Result<()> {
    for days_forward in 0..=config.premake_days {
        let day = context
            .today
            .checked_add_days(chrono::Days::new(days_forward as u64))
            .ok_or_else(|| eyre!("failed to calculate premake day"))?;
        let next_day = day
            .checked_add_days(chrono::Days::new(1))
            .ok_or_else(|| eyre!("failed to calculate premake next day"))?;
        let partition = daily_partition_name(table, day);
        let ddl = format!(
            "CREATE TABLE IF NOT EXISTS {} PARTITION OF {} FOR VALUES FROM ('{}') TO ('{}')",
            quote_identifier(&partition),
            quote_identifier(table),
            day_start_literal(day),
            day_start_literal(next_day)
        );

        execute_ddl(
            pool,
            config.statement_timeout,
            &ddl,
            table,
            &partition,
            "premake_partition",
            &context.party,
        )
        .await?;
    }

    Ok(())
}

async fn reap_range_partitions(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    table: &str,
) -> Result<()> {
    let cutoff = context.now - chrono::Duration::days(config.retention_days);
    let partitions = list_range_partitions(pool, config, table, &context.party).await?;

    for partition in partitions {
        if range_is_older_than_window(partition.upper_bound, cutoff) {
            detach_and_quarantine_partition(
                pool,
                config,
                context,
                table,
                &partition.name,
                "reap_range_partition",
            )
            .await?;
        } else {
            info!(
                table,
                partition = partition.name.as_str(),
                party_id = context.party.as_str(),
                operation = "reap_range_partition_skip",
                partition_upper_bound = %partition.upper_bound,
                cutoff = %cutoff,
                "partition is still within retention window"
            );
        }
    }

    Ok(())
}

async fn drop_expired_quarantine_relations(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    schema_name: &str,
    table: &str,
) -> Result<()> {
    let cutoff = context.now - chrono::Duration::hours(config.quarantine_grace_hours);
    let quarantine_relations =
        list_quarantine_relations(pool, config, table, schema_name, &context.party).await?;

    for relation in quarantine_relations {
        let detach_time = Utc.from_utc_datetime(
            &relation
                .detach_date
                .and_hms_opt(0, 0, 0)
                .ok_or_else(|| eyre!("failed to build quarantine detach timestamp"))?,
        );
        if detach_time < cutoff {
            let relation_exists =
                relation_exists(pool, config, &relation.name, table, &context.party).await?;
            if !relation_exists {
                warn!(
                    table,
                    partition = relation.name.as_str(),
                    party_id = context.party.as_str(),
                    operation = "drop_quarantine_skip_missing",
                    "quarantine relation vanished before drop"
                );
                continue;
            }

            let bytes_dropped = match relation_size_bytes(
                pool,
                config,
                &relation.name,
                table,
                &context.party,
            )
            .await
            {
                Ok(bytes_dropped) => bytes_dropped,
                Err(error) => {
                    warn!(
                        table,
                        partition = relation.name.as_str(),
                        party_id = context.party.as_str(),
                        operation = "relation_size_bytes_estimate_failed",
                        error = %error,
                        "failed to probe quarantine relation size; continuing with zero"
                    );
                    0
                }
            };
            let rows_dropped =
                match relation_row_count(pool, config, &relation.name, table, &context.party).await
                {
                    Ok(rows_dropped) => rows_dropped,
                    Err(error) => {
                        warn!(
                            table,
                            partition = relation.name.as_str(),
                            party_id = context.party.as_str(),
                            operation = "relation_row_count_estimate_failed",
                            error = %error,
                            "failed to probe quarantine relation row count; continuing with zero"
                        );
                        0
                    }
                };
            let ddl = format!("DROP TABLE IF EXISTS {}", quote_identifier(&relation.name));
            execute_ddl(
                pool,
                config.statement_timeout,
                &ddl,
                table,
                &relation.name,
                "drop_quarantine_relation",
                &context.party,
            )
            .await?;
            metrics::counter!(
                "anon_stats_retention.rows_dropped",
                "table" => table.to_string(),
                "party" => context.party.clone()
            )
            .increment(rows_dropped as u64);
            metrics::counter!(
                "anon_stats_retention.bytes_dropped",
                "table" => table.to_string(),
                "party" => context.party.clone()
            )
            .increment(bytes_dropped as u64);
        } else {
            info!(
                table,
                partition = relation.name.as_str(),
                party_id = context.party.as_str(),
                operation = "drop_quarantine_skip_grace",
                detach_time = %detach_time,
                cutoff = %cutoff,
                "quarantine relation is still inside grace period"
            );
        }
    }

    Ok(())
}

async fn finalize_pending_detach_partitions(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    party: &str,
) -> Result<()> {
    let partitions = list_pending_detach_partitions(pool, config, table, party).await?;

    for partition in partitions {
        finalize_pending_detach_partition(pool, config, table, &partition, party).await?;
    }

    Ok(())
}

async fn list_pending_detach_partitions(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    party: &str,
) -> Result<Vec<String>> {
    let sql = r#"
        SELECT c.relname
        FROM pg_inherits i
        JOIN pg_class p ON p.oid = i.inhparent
        JOIN pg_class c ON c.oid = i.inhrelid
        JOIN pg_namespace n ON n.oid = p.relnamespace
        WHERE p.oid = to_regclass($1)
          AND n.oid = current_schema()::regnamespace
          AND i.inhdetachpending = true
        ORDER BY c.relname
    "#;
    let rows = fetch_all_rows(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(table),
        table,
        "list_pending_detach_partitions",
        party,
    )
    .await?;

    rows.into_iter()
        .map(|row| {
            row.try_get("relname").with_context(|| {
                format!(
                    "table={table} operation=list_pending_detach_partitions_get_name party={party}"
                )
            })
        })
        .collect()
}

async fn finalize_pending_detach_partition(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    partition: &str,
    party: &str,
) -> Result<()> {
    let operation = "finalize_pending_detach";
    let mut connection = pool.acquire().await.with_context(|| {
        format!(
            "table={table} partition={partition} operation=acquire_finalize_connection party={party}"
        )
    })?;
    set_statement_timeout(&mut connection, config.detach_timeout)
        .await
        .with_context(|| {
            format!("table={table} partition={partition} operation=set_finalize_statement_timeout party={party}")
        })?;

    let ddl = finalize_detach_partition_ddl(table, partition);
    timeout(
        config.detach_timeout + Duration::from_secs(5),
        connection.execute(ddl.as_str()),
    )
    .await
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation}_timeout party={party}")
    })?
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation} party={party}")
    })?;

    info!(
        table,
        partition,
        party_id = party,
        operation,
        "finalized pending partition detach"
    );
    Ok(())
}

async fn reap_legacy_default_partition(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    table: &str,
) -> Result<()> {
    let legacy_partition = format!("{table}_legacy");
    let attached = partition_is_attached(pool, config, table, &legacy_partition, &context.party)
        .await
        .with_context(|| {
            format!(
                "table={table} partition={legacy_partition} operation=check_legacy_attached party={}",
                context.party
            )
        })?;
    if !attached {
        info!(
            table,
            partition = legacy_partition.as_str(),
            party_id = context.party.as_str(),
            operation = "reap_legacy_skip_missing",
            "legacy default partition is not attached"
        );
        return Ok(());
    }

    match legacy_should_reap(pool, config, table, &legacy_partition, &context.party).await? {
        Some(true) => {
            detach_and_quarantine_partition(
                pool,
                config,
                context,
                table,
                &legacy_partition,
                "reap_legacy_default",
            )
            .await?;
        }
        Some(false) => {
            let cutoff = context.now - chrono::Duration::days(config.retention_days);
            info!(
                table,
                partition = legacy_partition.as_str(),
                party_id = context.party.as_str(),
                operation = "reap_legacy_skip_retained",
                retention_days = config.retention_days,
                cutoff = %cutoff,
                "legacy default partition contains retained data"
            );
        }
        None => {
            info!(
                table,
                partition = legacy_partition.as_str(),
                party_id = context.party.as_str(),
                operation = "reap_legacy_skip_empty",
                "legacy default partition is empty"
            );
        }
    }

    Ok(())
}

async fn emit_live_partition_metrics(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    table: &str,
) -> Result<()> {
    let partitions = list_range_partitions(pool, config, table, &context.party).await?;
    metrics::gauge!(
        "anon_stats_retention.partition_count",
        "table" => table.to_string(),
        "party" => context.party.clone()
    )
    .set(partitions.len() as f64);

    if let Some(oldest_partition) = partitions
        .iter()
        .min_by_key(|partition| partition.lower_bound)
    {
        metrics::gauge!(
            "anon_stats_retention.oldest_retained_created_at",
            "table" => table.to_string(),
            "party" => context.party.clone()
        )
        .set(oldest_partition.lower_bound.timestamp() as f64);
    } else {
        warn!(
            table,
            party_id = context.party.as_str(),
            operation = "oldest_retained_created_at_unavailable",
            "no live range partitions found"
        );
    }

    // Buffer-health gauge for the partitions-ahead-below-premake alert: how many whole
    // days of forward daily partitions exist beyond today. A range partition's upper_bound
    // is the exclusive TO bound (start of the day after its last covered day), so the
    // newest partition's upper_bound.date() is the premake forward edge. Subtracting today
    // yields the buffer depth. Emitted signed and unclamped: premake keeps this positive,
    // and a drop to 0 or negative is precisely the condition the alert must fire on, before
    // an insert could ever miss a partition.
    if let Some(newest_partition) = partitions
        .iter()
        .max_by_key(|partition| partition.upper_bound)
    {
        let partitions_ahead_days =
            partitions_ahead_days(newest_partition.upper_bound, context.today);
        metrics::gauge!(
            "anon_stats_retention.partitions_ahead_days",
            "table" => table.to_string(),
            "party" => context.party.clone()
        )
        .set(partitions_ahead_days as f64);
    } else {
        warn!(
            table,
            party_id = context.party.as_str(),
            operation = "partitions_ahead_days_unavailable",
            "no live range partitions found"
        );
    }

    Ok(())
}

async fn detach_and_quarantine_partition(
    pool: &PgPool,
    config: &RetentionConfig,
    context: &RunContext,
    table: &str,
    partition: &str,
    operation: &str,
) -> Result<()> {
    let attached = partition_is_attached(pool, config, table, partition, &context.party).await?;
    if !attached {
        info!(
            table,
            partition,
            party_id = context.party.as_str(),
            operation,
            "partition is already detached"
        );
        return Ok(());
    }

    let quarantine_name = quarantine_partition_name(table, context.today);
    let rename_target =
        next_available_quarantine_name(pool, config, table, &quarantine_name, &context.party)
            .await?;

    detach_partition_concurrently(pool, config, table, partition, operation, &context.party)
        .await?;
    rename_detached_partition(
        pool,
        config,
        table,
        partition,
        &rename_target,
        operation,
        &context.party,
    )
    .await?;

    Ok(())
}

async fn detach_partition_concurrently(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    partition: &str,
    operation: &str,
    party: &str,
) -> Result<()> {
    let mut connection = pool.acquire().await.with_context(|| {
        format!(
            "table={table} partition={partition} operation=acquire_detach_connection party={party}"
        )
    })?;
    set_statement_timeout(&mut connection, config.detach_timeout)
        .await
        .with_context(|| {
            format!("table={table} partition={partition} operation=set_detach_statement_timeout party={party}")
        })?;

    let ddl = format!(
        "ALTER TABLE {} DETACH PARTITION {} CONCURRENTLY",
        quote_identifier(table),
        quote_identifier(partition)
    );
    timeout(
        config.detach_timeout + Duration::from_secs(5),
        connection.execute(ddl.as_str()),
    )
    .await
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation}_timeout party={party}")
    })?
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation} party={party}")
    })?;

    info!(
        table,
        partition,
        party_id = party,
        operation,
        "detached partition concurrently"
    );
    Ok(())
}

async fn rename_detached_partition(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    partition: &str,
    target: &str,
    operation: &str,
    party: &str,
) -> Result<()> {
    if !relation_exists(pool, config, partition, table, party).await? {
        warn!(
            table,
            partition,
            party_id = party,
            operation,
            "detached partition vanished before rename"
        );
        return Ok(());
    }

    let ddl = format!(
        "ALTER TABLE {} RENAME TO {}",
        quote_identifier(partition),
        quote_identifier(target)
    );
    execute_ddl(
        pool,
        config.statement_timeout,
        &ddl,
        table,
        partition,
        "rename_to_quarantine",
        party,
    )
    .await?;
    info!(
        table,
        partition,
        quarantine = target,
        party_id = party,
        operation,
        "renamed detached partition to quarantine relation"
    );
    Ok(())
}

async fn list_range_partitions(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    party: &str,
) -> Result<Vec<RangePartition>> {
    // Catalog-as-state: pg_inherits links parent oid to attached child relations.
    // pg_class.relpartbound is rendered through pg_get_expr(c.relpartbound, c.oid)
    // into strings like "FOR VALUES FROM ('2026-06-01 00:00:00+00') TO ('2026-06-02 00:00:00+00')".
    // DEFAULT bounds render as "DEFAULT" and are intentionally skipped here; the legacy default
    // partition is handled by the dedicated legacy path.
    let sql = r#"
        SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) AS partition_bound
        FROM pg_inherits i
        JOIN pg_class p ON p.oid = i.inhparent
        JOIN pg_class c ON c.oid = i.inhrelid
        JOIN pg_namespace n ON n.oid = p.relnamespace
        WHERE p.oid = to_regclass($1)
          AND n.oid = current_schema()::regnamespace
          AND i.inhdetachpending = false
        ORDER BY c.relname
    "#;
    let rows = fetch_all_rows(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(table),
        table,
        "list_range_partitions",
        party,
    )
    .await?;

    let mut partitions = Vec::new();
    for row in rows {
        let name: String = row.try_get("relname").with_context(|| {
            format!("table={table} operation=list_range_partitions_get_name party={party}")
        })?;
        let bound: Option<String> = row.try_get("partition_bound").with_context(|| {
            format!("table={table} operation=list_range_partitions_get_bound party={party}")
        })?;
        match bound.as_deref() {
            Some("DEFAULT") => {
                info!(
                    table,
                    partition = name.as_str(),
                    party_id = party,
                    operation = "list_range_partitions_skip_default",
                    "skipping default partition"
                );
            }
            Some(bound_text) => {
                let (lower_bound, upper_bound) = parse_partition_bounds(bound_text)
                    .with_context(|| {
                        format!(
                            "table={table} partition={name} operation=parse_partition_bounds party={party}"
                        )
                    })?;
                partitions.push(RangePartition {
                    name,
                    lower_bound,
                    upper_bound,
                });
            }
            None => {
                warn!(
                    table,
                    partition = name.as_str(),
                    party_id = party,
                    operation = "list_range_partitions_skip_missing_bound",
                    "skipping partition with missing relpartbound expression"
                );
            }
        }
    }

    Ok(partitions)
}

async fn list_quarantine_relations(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    schema_name: &str,
    party: &str,
) -> Result<Vec<QuarantineRelation>> {
    let pattern = format!("{table}_quarantine_%");
    let sql = r#"
        SELECT c.relname
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1
          AND c.relkind IN ('r', 'p')
          AND c.relname LIKE $2
        ORDER BY c.relname
    "#;
    let rows = fetch_all_rows(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(schema_name).bind(&pattern),
        table,
        "list_quarantine_relations",
        party,
    )
    .await?;

    let mut relations = Vec::new();
    for row in rows {
        let name: String = row.try_get("relname").with_context(|| {
            format!("table={table} operation=list_quarantine_relations_get_name party={party}")
        })?;
        if let Some(detach_date) = parse_quarantine_date(table, &name) {
            relations.push(QuarantineRelation { name, detach_date });
        } else {
            warn!(
                table,
                partition = name.as_str(),
                party_id = party,
                operation = "list_quarantine_relations_skip_unparseable",
                "skipping quarantine relation with unparseable date suffix"
            );
        }
    }

    Ok(relations)
}

async fn next_available_quarantine_name(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    base_name: &str,
    party: &str,
) -> Result<String> {
    if !relation_exists(pool, config, base_name, table, party).await? {
        return Ok(base_name.to_string());
    }

    for suffix in 1..=10_000 {
        let candidate = format!("{base_name}_{suffix}");
        if !relation_exists(pool, config, &candidate, table, party).await? {
            return Ok(candidate);
        }
    }

    Err(eyre!(
        "table={table} operation=next_available_quarantine_name party={party}: exhausted suffixes"
    ))
}

async fn partition_is_attached(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    partition: &str,
    party: &str,
) -> Result<bool> {
    let sql = r#"
        SELECT EXISTS (
            SELECT 1
            FROM pg_inherits i
            JOIN pg_class p ON p.oid = i.inhparent
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_namespace n ON n.oid = p.relnamespace
            WHERE p.oid = to_regclass($1)
              AND c.relname = $2
              AND n.oid = current_schema()::regnamespace
              AND i.inhdetachpending = false
        ) AS exists
    "#;
    fetch_exists(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(table).bind(partition),
        table,
        partition,
        "partition_is_attached",
        party,
    )
    .await
}

async fn relation_exists(
    pool: &PgPool,
    config: &RetentionConfig,
    relation: &str,
    table: &str,
    party: &str,
) -> Result<bool> {
    let sql = "SELECT to_regclass($1) IS NOT NULL AS exists";
    fetch_exists(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(relation),
        table,
        relation,
        "relation_exists",
        party,
    )
    .await
}

async fn relation_size_bytes(
    pool: &PgPool,
    config: &RetentionConfig,
    relation: &str,
    table: &str,
    party: &str,
) -> Result<i64> {
    let sql = "SELECT COALESCE(pg_total_relation_size(to_regclass($1)), 0)::bigint AS size_bytes";
    let row = fetch_one_row(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(relation),
        table,
        relation,
        "relation_size_bytes",
        party,
    )
    .await?;
    row.try_get("size_bytes").with_context(|| {
        format!(
            "table={table} partition={relation} operation=relation_size_bytes_decode party={party}"
        )
    })
}

async fn relation_row_count(
    pool: &PgPool,
    config: &RetentionConfig,
    relation: &str,
    table: &str,
    party: &str,
) -> Result<i64> {
    let sql = "SELECT reltuples::bigint AS row_count FROM pg_class WHERE oid = to_regclass($1)";
    let row = fetch_one_row(
        pool,
        config.statement_timeout,
        sqlx::query(sql).bind(relation),
        table,
        relation,
        "relation_row_count",
        party,
    )
    .await?;
    row.try_get("row_count").with_context(|| {
        format!(
            "table={table} partition={relation} operation=relation_row_count_decode party={party}"
        )
    })
}

async fn legacy_should_reap(
    pool: &PgPool,
    config: &RetentionConfig,
    table: &str,
    legacy_partition: &str,
    party: &str,
) -> Result<Option<bool>> {
    let retention_days = i32::try_from(config.retention_days).with_context(|| {
        format!(
            "table={table} partition={legacy_partition} operation=legacy_retention_days_convert party={party}"
        )
    })?;
    let sql = format!(
        "SELECT (max(created_at) < now() - make_interval(days => $1)) AS should_reap \
         FROM {} HAVING max(created_at) IS NOT NULL",
        quote_identifier(legacy_partition)
    );
    let row = timeout(
        config.statement_timeout + Duration::from_secs(5),
        sqlx::query(&sql)
            .bind(retention_days)
            .fetch_optional(pool),
    )
    .await
    .with_context(|| {
        format!(
            "table={table} partition={legacy_partition} operation=legacy_should_reap_timeout party={party}"
        )
    })?
    .with_context(|| {
        format!("table={table} partition={legacy_partition} operation=legacy_should_reap party={party}")
    })?;
    row.map(|row| {
        row.try_get("should_reap").with_context(|| {
            format!("table={table} partition={legacy_partition} operation=legacy_should_reap_decode party={party}")
        })
    })
    .transpose()
}

async fn execute_ddl(
    pool: &PgPool,
    statement_timeout: Duration,
    sql: &str,
    table: &str,
    partition: &str,
    operation: &str,
    party: &str,
) -> Result<()> {
    let mut connection = pool.acquire().await.with_context(|| {
        format!("table={table} partition={partition} operation=acquire_connection party={party}")
    })?;
    set_statement_timeout(&mut connection, statement_timeout)
        .await
        .with_context(|| {
            format!(
                "table={table} partition={partition} operation=set_statement_timeout party={party}"
            )
        })?;
    timeout(
        statement_timeout + Duration::from_secs(5),
        connection.execute(sql),
    )
    .await
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation}_timeout party={party}")
    })?
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation} party={party}")
    })?;

    info!(
        table,
        partition,
        party_id = party,
        operation,
        "executed DDL"
    );
    Ok(())
}

async fn set_statement_timeout(
    connection: &mut PgConnection,
    statement_timeout: Duration,
) -> Result<()> {
    let timeout_sql = format!(
        "SET statement_timeout = '{}s'",
        statement_timeout.as_secs().max(1)
    );
    connection.execute(timeout_sql.as_str()).await?;
    Ok(())
}

async fn fetch_one_row<'q>(
    pool: &PgPool,
    statement_timeout: Duration,
    query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    table: &str,
    partition: &str,
    operation: &str,
    party: &str,
) -> Result<PgRow> {
    timeout(
        statement_timeout + Duration::from_secs(5),
        query.fetch_one(pool),
    )
    .await
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation}_timeout party={party}")
    })?
    .with_context(|| {
        format!("table={table} partition={partition} operation={operation} party={party}")
    })
}

async fn fetch_all_rows<'q>(
    pool: &PgPool,
    statement_timeout: Duration,
    query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    table: &str,
    operation: &str,
    party: &str,
) -> Result<Vec<PgRow>> {
    timeout(
        statement_timeout + Duration::from_secs(5),
        query.fetch_all(pool),
    )
    .await
    .with_context(|| format!("table={table} operation={operation}_timeout party={party}"))?
    .with_context(|| format!("table={table} operation={operation} party={party}"))
}

async fn fetch_exists<'q>(
    pool: &PgPool,
    statement_timeout: Duration,
    query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    table: &str,
    partition: &str,
    operation: &str,
    party: &str,
) -> Result<bool> {
    let row = fetch_one_row(
        pool,
        statement_timeout,
        query,
        table,
        partition,
        operation,
        party,
    )
    .await?;
    row.try_get("exists").with_context(|| {
        format!("table={table} partition={partition} operation={operation}_decode party={party}")
    })
}

impl RetentionConfig {
    fn from_env() -> Result<Self> {
        let tables = env::var("ANON_STATS_RETENTION_TABLES")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
            .filter(|tables| !tables.is_empty())
            .unwrap_or_else(|| DEFAULT_TABLES.iter().map(ToString::to_string).collect());

        for table in &tables {
            validate_identifier(table)
                .with_context(|| format!("invalid ANON_STATS_RETENTION_TABLES entry: {table}"))?;
        }

        Ok(Self {
            tables,
            retention_days: parse_env_i64("RETENTION_DAYS", DEFAULT_RETENTION_DAYS)?,
            premake_days: parse_env_i64("PREMAKE_DAYS", DEFAULT_PREMAKE_DAYS)?,
            quarantine_grace_hours: parse_env_i64(
                "QUARANTINE_GRACE_HOURS",
                DEFAULT_QUARANTINE_GRACE_HOURS,
            )?,
            statement_timeout: Duration::from_secs(parse_env_u64(
                "STATEMENT_TIMEOUT_SECS",
                DEFAULT_STATEMENT_TIMEOUT_SECS,
            )?),
            detach_timeout: Duration::from_secs(parse_env_u64(
                "DETACH_TIMEOUT_SECS",
                DEFAULT_DETACH_TIMEOUT_SECS,
            )?),
        })
    }
}

fn parse_env_i64(name: &str, default: i64) -> Result<i64> {
    match env::var(name).ok() {
        Some(value) => value
            .parse::<i64>()
            .with_context(|| format!("failed to parse {name}={value} as i64")),
        None => Ok(default),
    }
}

fn parse_env_u64(name: &str, default: u64) -> Result<u64> {
    match env::var(name).ok() {
        Some(value) => value
            .parse::<u64>()
            .with_context(|| format!("failed to parse {name}={value} as u64")),
        None => Ok(default),
    }
}

fn daily_partition_name(table: &str, day: NaiveDate) -> String {
    format!("{table}_p{}", day.format("%Y%m%d"))
}

fn quarantine_partition_name(table: &str, day: NaiveDate) -> String {
    format!("{table}_quarantine_{}", day.format("%Y%m%d"))
}

fn day_start_literal(day: NaiveDate) -> String {
    format!("{} 00:00:00+00", day.format("%Y-%m-%d"))
}

fn range_is_older_than_window(upper_bound: DateTime<Utc>, cutoff: DateTime<Utc>) -> bool {
    upper_bound <= cutoff
}

/// Whole-day buffer depth for the partitions-ahead-below-premake alert: the UTC date of the
/// newest forward partition's exclusive upper bound minus today. Signed and unclamped so the
/// alert can fire when the premake buffer erodes to zero or goes negative.
fn partitions_ahead_days(newest_upper_bound: DateTime<Utc>, today: NaiveDate) -> i64 {
    (newest_upper_bound.date_naive() - today).num_days()
}

fn parse_quarantine_date(table: &str, relname: &str) -> Option<NaiveDate> {
    let prefix = format!("{table}_quarantine_");
    let suffix = relname.strip_prefix(&prefix)?;
    let date_part = suffix.split('_').next()?;
    if date_part.len() != 8 {
        return None;
    }
    NaiveDate::parse_from_str(date_part, "%Y%m%d").ok()
}

fn parse_partition_bounds(bound: &str) -> Result<(DateTime<Utc>, DateTime<Utc>)> {
    let from_marker = "FROM (";
    let to_marker = ") TO (";
    let from_start = bound
        .find(from_marker)
        .ok_or_else(|| eyre!("partition bound missing FROM marker: {bound}"))?
        + from_marker.len();
    let to_start = bound
        .find(to_marker)
        .ok_or_else(|| eyre!("partition bound missing TO marker: {bound}"))?;
    let to_value_start = to_start + to_marker.len();
    let to_end = bound
        .rfind(')')
        .ok_or_else(|| eyre!("partition bound missing closing paren: {bound}"))?;

    let lower_text = strip_bound_quotes(&bound[from_start..to_start]);
    let upper_text = strip_bound_quotes(&bound[to_value_start..to_end]);
    let lower_bound = parse_partition_bound_timestamp(lower_text)?;
    let upper_bound = parse_partition_bound_timestamp(upper_text)?;
    Ok((lower_bound, upper_bound))
}

fn strip_bound_quotes(value: &str) -> &str {
    value
        .trim()
        .trim_start_matches('\'')
        .trim_end_matches('\'')
        .trim()
}

fn parse_partition_bound_timestamp(value: &str) -> Result<DateTime<Utc>> {
    let normalized = if value.ends_with("+00") {
        format!("{value}:00")
    } else {
        value.to_string()
    };
    DateTime::parse_from_str(&normalized, "%Y-%m-%d %H:%M:%S%:z")
        .or_else(|_| DateTime::parse_from_rfc3339(&normalized))
        .map(|value| value.with_timezone(&Utc))
        .with_context(|| format!("failed to parse partition timestamp bound: {value}"))
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn finalize_detach_partition_ddl(table: &str, partition: &str) -> String {
    format!(
        "ALTER TABLE {} DETACH PARTITION {} FINALIZE",
        quote_identifier(table),
        quote_identifier(partition)
    )
}

fn validate_identifier(identifier: &str) -> Result<()> {
    let mut chars = identifier.chars();
    match chars.next() {
        Some(first) if first == '_' || first.is_ascii_alphabetic() => {}
        Some(_) | None => {
            return Err(eyre!(
                "identifier must start with an ASCII letter or underscore"
            ))
        }
    }

    for character in chars {
        if !(character == '_' || character.is_ascii_alphanumeric()) {
            return Err(eyre!(
                "identifier contains non-ASCII-alphanumeric character: {identifier}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(year: i32, month: u32, day: u32, hour: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, 0, 0)
            .single()
            .expect("test timestamp should be valid")
    }

    #[test]
    fn daily_partition_name_uses_utc_day() {
        let day = NaiveDate::from_ymd_opt(2026, 6, 1).expect("test date should be valid");
        assert_eq!(
            daily_partition_name("anon_stats_1d", day),
            "anon_stats_1d_p20260601"
        );
    }

    #[test]
    fn quarantine_partition_name_uses_detach_day() {
        let day = NaiveDate::from_ymd_opt(2026, 6, 30).expect("test date should be valid");
        assert_eq!(
            quarantine_partition_name("anon_stats_face", day),
            "anon_stats_face_quarantine_20260630"
        );
    }

    #[test]
    fn range_is_old_when_upper_bound_is_at_cutoff() {
        let cutoff = utc(2026, 6, 16, 12);
        assert!(range_is_older_than_window(cutoff, cutoff));
    }

    #[test]
    fn range_is_retained_when_upper_bound_is_after_cutoff() {
        let cutoff = utc(2026, 6, 16, 12);
        let upper_bound = utc(2026, 6, 16, 13);
        assert!(!range_is_older_than_window(upper_bound, cutoff));
    }

    #[test]
    fn parses_quarantine_date_without_suffix() {
        let date = parse_quarantine_date("anon_stats_1d", "anon_stats_1d_quarantine_20260630")
            .expect("quarantine date should parse");
        assert_eq!(
            date,
            NaiveDate::from_ymd_opt(2026, 6, 30).expect("test date should be valid")
        );
    }

    #[test]
    fn parses_quarantine_date_with_collision_suffix() {
        let date = parse_quarantine_date("anon_stats_1d", "anon_stats_1d_quarantine_20260630_2")
            .expect("quarantine date should parse");
        assert_eq!(
            date,
            NaiveDate::from_ymd_opt(2026, 6, 30).expect("test date should be valid")
        );
    }

    #[test]
    fn rejects_unparseable_quarantine_date() {
        assert!(parse_quarantine_date("anon_stats_1d", "anon_stats_1d_quarantine_today").is_none());
    }

    #[test]
    fn partitions_ahead_days_counts_forward_buffer() {
        // Newest partition's exclusive upper bound is 2026-06-30 (start of day), today is
        // 2026-06-25: five whole days of forward buffer remain.
        let today = NaiveDate::from_ymd_opt(2026, 6, 25).expect("test date should be valid");
        assert_eq!(partitions_ahead_days(utc(2026, 6, 30, 0), today), 5);
    }

    #[test]
    fn partitions_ahead_days_is_zero_at_today_edge() {
        // Upper bound equals start of today: the buffer is exhausted — alert-firing condition.
        let today = NaiveDate::from_ymd_opt(2026, 6, 25).expect("test date should be valid");
        assert_eq!(partitions_ahead_days(utc(2026, 6, 25, 0), today), 0);
    }

    #[test]
    fn partitions_ahead_days_goes_negative_when_buffer_eroded() {
        // Newest upper bound is before today: premake has fallen behind, signed negative.
        let today = NaiveDate::from_ymd_opt(2026, 6, 25).expect("test date should be valid");
        assert_eq!(partitions_ahead_days(utc(2026, 6, 23, 0), today), -2);
    }

    #[test]
    fn partitions_ahead_days_ignores_intraday_time() {
        // A non-midnight upper bound still maps to its UTC calendar day for the delta.
        let today = NaiveDate::from_ymd_opt(2026, 6, 25).expect("test date should be valid");
        assert_eq!(partitions_ahead_days(utc(2026, 6, 28, 13), today), 3);
    }

    #[test]
    fn parses_partition_bounds_from_relpartbound_expression() {
        let bound = "FOR VALUES FROM ('2026-06-01 00:00:00+00') TO ('2026-06-02 00:00:00+00')";
        let (lower, upper) = parse_partition_bounds(bound).expect("partition bounds should parse");
        assert_eq!(lower, utc(2026, 6, 1, 0));
        assert_eq!(upper, utc(2026, 6, 2, 0));
    }

    #[test]
    fn parses_partition_bounds_with_rfc3339_timestamps() {
        let bound = "FOR VALUES FROM ('2026-06-01T00:00:00Z') TO ('2026-06-02T00:00:00Z')";
        let (lower, upper) = parse_partition_bounds(bound).expect("partition bounds should parse");
        assert_eq!(lower, utc(2026, 6, 1, 0));
        assert_eq!(upper, utc(2026, 6, 2, 0));
    }

    #[test]
    fn finalize_detach_partition_ddl_quotes_parent_and_child() {
        assert_eq!(
            finalize_detach_partition_ddl("anon_stats_1d", "anon_stats_1d_p20260601"),
            "ALTER TABLE \"anon_stats_1d\" DETACH PARTITION \"anon_stats_1d_p20260601\" FINALIZE"
        );
    }

    #[test]
    fn validates_expected_table_identifier() {
        validate_identifier("anon_stats_1d_lifted").expect("identifier should be valid");
    }

    #[test]
    fn rejects_identifier_with_dot() {
        assert!(validate_identifier("public.anon_stats_1d").is_err());
    }
}
