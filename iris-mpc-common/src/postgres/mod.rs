use eyre::{bail, eyre, Result, WrapErr};
use sqlx::{postgres::PgPoolOptions, Executor, PgPool};
use std::time::Duration;

const MAX_CONNECTIONS: u32 = 100;
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub struct PostgresClient {
    pub pool: PgPool,
    pub schema_name: String,
}

// Helper type: name of a PostgreSQL schema.
pub type PostgresSchemaName = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Select an existing schema without restricting writes.
    ExistingSchema,
    /// Skip schema creation and reject writes on every pooled connection.
    EnforcedReadOnly,
    ReadWrite,
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if !input.is_empty()
        && input.len() <= 63
        && input
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
    {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

fn sql_switch_schema(schema_name: &str, access_mode: AccessMode) -> Result<String> {
    sanitize_identifier(schema_name)?;

    if access_mode == AccessMode::EnforcedReadOnly {
        Ok(format!(
            "SET default_transaction_read_only = on; SET search_path TO \"{}\";",
            schema_name
        ))
    } else if access_mode == AccessMode::ExistingSchema {
        Ok(format!("SET search_path TO \"{}\";", schema_name))
    } else {
        Ok(format!(
            "
            CREATE SCHEMA IF NOT EXISTS \"{}\";
            SET search_path TO \"{}\", public;
            ",
            schema_name, schema_name
        ))
    }
}

pub async fn is_cpu_cell_managed(pool: &PgPool, schema: &str) -> Result<bool> {
    sanitize_identifier(schema)?;
    tokio::time::timeout(
        HANDOFF_TIMEOUT,
        sqlx::query_scalar(&format!(
            "SELECT EXISTS (SELECT 1 FROM \"{schema}\".cpu_cell_handoffs)"
        ))
        .fetch_one(pool),
    )
    .await
    .wrap_err("CPU handoff lookup timed out")?
    .map_err(Into::into)
}

/// Check after migrations, before replay, queue access or operational writes. A recorded
/// activation cannot authorize a fresh process with none of B's resident state.
/// Locking the row waits for an old process's in-flight drain commit before
/// deciding whether its replacement can start.
pub async fn check_handoff_startup(
    pool: &PgPool,
    schema: &str,
    deployment: &str,
    candidate: bool,
) -> Result<()> {
    sanitize_identifier(schema)?;
    let deployment = uuid::Uuid::parse_str(deployment).wrap_err("invalid CPU deployment UUID")?;
    let lock = if candidate { "" } else { "FOR SHARE" };
    let current: Option<(uuid::Uuid, uuid::Uuid, String)> = tokio::time::timeout(
        HANDOFF_TIMEOUT,
        sqlx::query_as(&format!(
            "SELECT source, destination, phase FROM \"{schema}\".cpu_cell_handoffs
             ORDER BY sequence DESC LIMIT 1 {lock}"
        ))
        .fetch_optional(pool),
    )
    .await
    .wrap_err("CPU handoff startup check timed out")??;
    let allowed = current.is_some_and(|(source, destination, phase)| {
        if candidate {
            destination == deployment
                && matches!(phase.as_str(), "requested" | "draining" | "drained")
        } else {
            source == deployment && phase == "requested"
        }
    });
    if !allowed {
        bail!("CPU deployment cannot start in this handoff phase; coordinated cold recovery is required after an interrupted handoff");
    }
    Ok(())
}

impl PostgresClient {
    pub async fn new(url: &str, schema_name: &str, access_mode: AccessMode) -> Result<Self> {
        tracing::info!("Connecting to V2 database with, schema: {}", schema_name);
        let connect_sql = sql_switch_schema(schema_name, access_mode)?;

        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .after_connect(move |conn, _meta| {
                // Switch to the given schema in every connection.
                let connect_sql = connect_sql.clone();
                Box::pin(async move {
                    conn.execute(connect_sql.as_ref()).await.inspect_err(|e| {
                        tracing::error!("error in after_connect: {:?}", e);
                    })?;
                    Ok(())
                })
            })
            .connect(url)
            .await?;

        Ok(PostgresClient {
            pool,
            schema_name: schema_name.to_string(),
        })
    }
}

/// Runs pending database migrations against the schema associated with `client`.
///
/// When `ignore_missing_migrations` is true,
/// migrations applied to the database but missing from the local migrations
/// directory are ignored instead of causing the migration to fail.
///
/// This is intentionally decoupled from connection/store construction so that
/// migrations run explicitly, once, at application startup.
pub async fn run_migrations(pool: &PgPool, ignore_missing_migrations: bool) -> Result<()> {
    tracing::info!(
        "Running migrations (ignore_missing_migrations={})...",
        ignore_missing_migrations
    );

    sqlx::migrate!("./../migrations")
        .set_ignore_missing(ignore_missing_migrations)
        .run(pool)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires CPU_CELL_TEST_DATABASE_URL pointing to disposable PostgreSQL"]
    async fn handoff_migration_is_inert_until_requested_and_retains_history() -> Result<()> {
        let url = std::env::var("CPU_CELL_TEST_DATABASE_URL")?;
        let schema = format!("upgrade_{}", uuid::Uuid::new_v4().simple());
        let client = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
        run_migrations(&client.pool, false).await?;
        run_migrations(&client.pool, false).await?;
        assert!(!is_cpu_cell_managed(&client.pool, &schema).await?);
        let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
        run_migrations(&ordinary.pool, false).await?;
        ordinary.pool.close().await;
        sqlx::query(
            "INSERT INTO cpu_cell_handoffs (handoff_id, source, destination) VALUES ($1,$2,$3)",
        )
        .bind(uuid::Uuid::new_v4())
        .bind(uuid::Uuid::new_v4())
        .bind(uuid::Uuid::new_v4())
        .execute(&client.pool)
        .await?;
        assert!(is_cpu_cell_managed(&client.pool, &schema).await?);
        let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
        run_migrations(&ordinary.pool, false).await?;
        ordinary.pool.close().await;
        let count: i64 = sqlx::query_scalar("SELECT count(*) FROM cpu_cell_handoffs")
            .fetch_one(&client.pool)
            .await?;
        assert_eq!(count, 1);
        client
            .pool
            .execute(format!("DROP SCHEMA {schema} CASCADE").as_str())
            .await?;
        client.pool.close().await;
        Ok(())
    }

    #[tokio::test]
    async fn handoff_lookup_failure_is_not_legacy_mode() -> Result<()> {
        let pool =
            PgPoolOptions::new().connect_lazy("postgres://unused:unused@127.0.0.1:1/unused")?;
        pool.close().await;
        assert!(is_cpu_cell_managed(&pool, "canonical").await.is_err());
        assert!(PostgresClient::new(
            "postgres://unused:unused@127.0.0.1:1/unused",
            "canonical",
            AccessMode::ReadWrite
        )
        .await
        .is_err());
        Ok(())
    }
}
