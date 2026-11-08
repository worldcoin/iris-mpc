use eyre::{bail, eyre, Result, WrapErr};
use sqlx::{postgres::PgPoolOptions, Executor, PgConnection, PgPool};
use std::time::Duration;

const MAX_CONNECTIONS: u32 = 100;
const HANDOFF_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub struct PostgresClient {
    pub pool: PgPool,
    pub schema_name: String,
}

pub type PostgresSchemaName = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ExistingSchema,
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

fn sql_switch_schema(schema: &str, mode: AccessMode) -> Result<String> {
    sanitize_identifier(schema)?;
    Ok(match mode {
        AccessMode::EnforcedReadOnly => {
            format!("SET default_transaction_read_only = on; SET search_path TO \"{schema}\";")
        }
        AccessMode::ExistingSchema => format!("SET search_path TO \"{schema}\";"),
        AccessMode::ReadWrite => format!(
            "CREATE SCHEMA IF NOT EXISTS \"{schema}\"; SET search_path TO \"{schema}\", public;"
        ),
    })
}

async fn managed_connection(conn: &mut PgConnection, schema: &str) -> Result<bool, sqlx::Error> {
    sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=$1 AND c.relname='cpu_cell_handoffs')")
        .bind(schema).fetch_one(conn).await
}

pub async fn is_cpu_cell_managed(pool: &PgPool, schema: &str) -> Result<bool> {
    sanitize_identifier(schema)?;
    tokio::time::timeout(HANDOFF_TIMEOUT, async {
        let mut conn = pool.acquire().await?;
        managed_connection(&mut conn, schema).await
    })
    .await
    .wrap_err("CPU handoff lookup timed out")?
    .map_err(Into::into)
}

/// Check before migrations, replay, queue access or canonical writes. A recorded
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
    pub async fn new(url: &str, schema: &str, mode: AccessMode) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema, mode)?;
        let control = Self::new_control(url, schema).await?;
        if is_cpu_cell_managed(&control.pool, schema).await? && mode != AccessMode::EnforcedReadOnly
        {
            bail!("CPU handoff schema requires deployment-aware startup");
        }
        control.pool.close().await;
        let schema_name = schema.to_owned();
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .after_connect(move |conn, _| {
                let sql = connect_sql.clone();
                let schema = schema_name.clone();
                Box::pin(async move {
                    tokio::time::timeout(HANDOFF_TIMEOUT, async {
                        if mode != AccessMode::EnforcedReadOnly
                            && managed_connection(conn, &schema).await?
                        {
                            return Err(sqlx::Error::Protocol(
                                "CPU handoff schema requires deployment-aware startup".into(),
                            ));
                        }
                        conn.execute(sql.as_str()).await?;
                        Ok(())
                    })
                    .await
                    .map_err(|_| {
                        sqlx::Error::Protocol("CPU handoff startup guard timed out".into())
                    })?
                })
            })
            .connect(url)
            .await?;
        Ok(Self {
            pool,
            schema_name: schema.to_owned(),
        })
    }

    /// A small pool for ordinary handoff SQL; never creates schemas or runs migrations.
    pub async fn new_control(url: &str, schema: &str) -> Result<Self> {
        sanitize_identifier(schema)?;
        let pool = PgPoolOptions::new().max_connections(2).acquire_timeout(HANDOFF_TIMEOUT)
            .after_connect(|conn, _| Box::pin(async move {
                conn.execute("SET statement_timeout='5s'; SET lock_timeout='2s'; SET search_path TO pg_catalog;").await?;
                Ok(())
            })).connect(url).await?;
        Ok(Self {
            pool,
            schema_name: schema.to_owned(),
        })
    }

    /// The original A can reconnect to settle admitted work while draining.
    /// New processes must first pass the stricter requested-phase startup check.
    pub async fn new_handoff_source(url: &str, schema: &str, deployment: &str) -> Result<Self> {
        let control = Self::new_control(url, schema).await?;
        check_handoff_startup(&control.pool, schema, deployment, false).await?;
        control.pool.close().await;
        let deployment = uuid::Uuid::parse_str(deployment)?;
        let sql = sql_switch_schema(schema, AccessMode::ExistingSchema)?;
        let check = format!("SELECT COALESCE((SELECT source=$1 AND phase IN ('requested','draining') FROM \"{schema}\".cpu_cell_handoffs ORDER BY sequence DESC LIMIT 1), false)");
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .after_connect(move |conn, _| {
                let sql = sql.clone();
                let check = check.clone();
                Box::pin(async move {
                    tokio::time::timeout(HANDOFF_TIMEOUT, async {
                        let allowed: bool = sqlx::query_scalar(&check)
                            .bind(deployment)
                            .fetch_one(&mut *conn)
                            .await?;
                        if !allowed {
                            return Err(sqlx::Error::Protocol(
                                "CPU source has completed its handoff".into(),
                            ));
                        }
                        conn.execute(sql.as_str()).await?;
                        Ok(())
                    })
                    .await
                    .map_err(|_| {
                        sqlx::Error::Protocol("CPU handoff reconnect check timed out".into())
                    })?
                })
            })
            .connect(url)
            .await?;
        Ok(Self {
            pool,
            schema_name: schema.to_owned(),
        })
    }
}

pub async fn run_migrations(pool: &PgPool, ignore_missing_migrations: bool) -> Result<()> {
    let schema: String = sqlx::query_scalar("SELECT current_schema()")
        .fetch_one(pool)
        .await?;
    if is_cpu_cell_managed(pool, &schema).await? {
        bail!("managed CPU-cell schema requires separately provisioned migrations");
    }
    sqlx::migrate!("./../migrations")
        .set_ignore_missing(ignore_missing_migrations)
        .run(pool)
        .await?;
    Ok(())
}

/// Validate preprovisioned history without sqlx's migration-table creation or
/// advisory locking. All local migrations must already be applied unchanged.
pub async fn validate_migrations(pool: &PgPool, ignore_missing: bool) -> Result<()> {
    let applied: Vec<(i64, bool, Vec<u8>)> = tokio::time::timeout(
        HANDOFF_TIMEOUT,
        sqlx::query_as("SELECT version, success, checksum FROM _sqlx_migrations ORDER BY version")
            .fetch_all(pool),
    )
    .await
    .wrap_err("CPU-cell migration validation timed out")??;
    let migrator = sqlx::migrate!("./../migrations");
    for (version, success, checksum) in &applied {
        if !success {
            bail!("CPU-cell schema has failed migration {version}");
        }
        match migrator.iter().find(|migration| {
            migration.version == *version && migration.migration_type.is_up_migration()
        }) {
            Some(migration) if migration.checksum.as_ref() != checksum.as_slice() => {
                bail!("CPU-cell migration checksum mismatch at {version}")
            }
            None if !ignore_missing => bail!("CPU-cell unknown applied migration {version}"),
            _ => {}
        }
    }
    for migration in migrator
        .iter()
        .filter(|migration| migration.migration_type.is_up_migration())
    {
        if !applied
            .iter()
            .any(|(version, _, _)| *version == migration.version)
        {
            bail!(
                "CPU-cell schema requires deployment migration {} before startup",
                migration.version
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    #[ignore = "requires POP4453_TEST_DATABASE_URL pointing to disposable PostgreSQL"]
    async fn managed_migration_validation_checks_history_without_ddl() -> Result<()> {
        let url = std::env::var("POP4453_TEST_DATABASE_URL")?;
        let schema = format!("history_{}", uuid::Uuid::new_v4().simple());
        // Provisioning credentials remain usable after the authority is installed.
        let search_path = format!("SET search_path TO {schema}");
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .after_connect(move |conn, _| {
                let sql = search_path.clone();
                Box::pin(async move {
                    conn.execute(sql.as_str()).await?;
                    Ok(())
                })
            })
            .connect(&url)
            .await?;
        pool.execute(format!("CREATE SCHEMA {schema}").as_str())
            .await?;
        let client = PostgresClient {
            pool,
            schema_name: schema.clone(),
        };
        client
            .pool
            .execute("CREATE TABLE cpu_cell_handoffs (id int)")
            .await?;
        assert!(run_migrations(&client.pool, false).await.is_err());
        assert!(validate_migrations(&client.pool, false).await.is_err());
        let absent: bool = sqlx::query_scalar("SELECT to_regclass('_sqlx_migrations') IS NULL")
            .fetch_one(&client.pool)
            .await?;
        assert!(absent, "validation must not create migration history");
        client.pool.execute("CREATE TABLE _sqlx_migrations(version bigint PRIMARY KEY, success boolean, checksum bytea)").await?;
        let migrator = sqlx::migrate!("./../migrations");
        for migration in migrator
            .iter()
            .filter(|migration| migration.migration_type.is_up_migration())
        {
            sqlx::query("INSERT INTO _sqlx_migrations VALUES($1,true,$2)")
                .bind(migration.version)
                .bind(migration.checksum.as_ref())
                .execute(&client.pool)
                .await?;
        }
        validate_migrations(&client.pool, false).await?;
        client.pool.execute("UPDATE _sqlx_migrations SET checksum='bad'::bytea WHERE version=(SELECT min(version) FROM _sqlx_migrations)").await?;
        assert!(validate_migrations(&client.pool, false).await.is_err());
        client.pool.execute("DELETE FROM _sqlx_migrations WHERE version=(SELECT min(version) FROM _sqlx_migrations)").await?;
        assert!(
            validate_migrations(&client.pool, true).await.is_err(),
            "ignore_missing must not permit pending local migrations"
        );
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
