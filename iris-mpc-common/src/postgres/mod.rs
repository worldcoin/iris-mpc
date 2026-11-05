use eyre::{bail, eyre, Result, WrapErr};
use sqlx::{postgres::PgPoolOptions, Executor, PgConnection, PgPool};
use std::time::Duration;

const MAX_CONNECTIONS: u32 = 100;
const AUTHORITY_TIMEOUT: Duration = Duration::from_secs(10);

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
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
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
            "SET default_transaction_read_only = on; SET search_path TO \"{schema_name}\";"
        ))
    } else if access_mode == AccessMode::ExistingSchema {
        Ok(format!("SET search_path TO \"{schema_name}\";"))
    } else {
        Ok(format!("CREATE SCHEMA IF NOT EXISTS \"{schema_name}\"; SET search_path TO \"{schema_name}\", public;"))
    }
}

async fn managed_connection(conn: &mut PgConnection, schema: &str) -> Result<bool, sqlx::Error> {
    // pg_class is independent of search_path and table SELECT permissions. An
    // unreadable or malformed authority cannot silently become legacy mode.
    sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = 'cpu_cell_authority')")
        .bind(schema).fetch_one(conn).await
}

pub async fn is_cpu_cell_managed(pool: &PgPool, schema: &str) -> Result<bool> {
    sanitize_identifier(schema)?;
    tokio::time::timeout(AUTHORITY_TIMEOUT, async {
        let mut conn = pool.acquire().await?;
        managed_connection(&mut conn, schema).await
    })
    .await
    .wrap_err("CPU-cell authority lookup timed out")?
    .map_err(Into::into)
}

async fn require_select_only(conn: &mut PgConnection, schema: &str) -> Result<(), sqlx::Error> {
    let writable: bool = sqlx::query_scalar(
        r#"SELECT
          EXISTS (SELECT 1 FROM pg_catalog.pg_roles r WHERE r.rolname=current_user
            AND (r.rolsuper OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication OR r.rolbypassrls
              OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members WHERE member=r.oid)))
          OR EXISTS (SELECT 1 FROM pg_catalog.pg_namespace n WHERE n.nspname=$1
            AND (pg_catalog.has_schema_privilege(current_user,n.oid,'CREATE')
              OR pg_catalog.pg_has_role(current_user,n.nspowner,'USAGE')))
          OR EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=$1 AND c.relname NOT LIKE 'cpu_cell_%' AND CASE
              WHEN c.relkind IN ('r','p','v','m','f') THEN
                NOT pg_catalog.has_table_privilege(current_user,c.oid,'SELECT')
                OR pg_catalog.has_table_privilege(current_user,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
                OR pg_catalog.has_any_column_privilege(current_user,c.oid,'INSERT,UPDATE,REFERENCES')
                OR pg_catalog.pg_has_role(current_user,c.relowner,'USAGE')
              WHEN c.relkind='S' THEN pg_catalog.has_sequence_privilege(current_user,c.oid,'USAGE,UPDATE')
              ELSE false END)"#
    ).bind(schema).fetch_one(conn).await?;
    if writable {
        return Err(sqlx::Error::Protocol(
            "managed candidate requires SELECT-only canonical database privileges".into(),
        ));
    }
    Ok(())
}

impl PostgresClient {
    pub async fn new(url: &str, schema_name: &str, access_mode: AccessMode) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema_name, access_mode)?;
        let control = Self::new_control(url, schema_name).await?;
        let mut probe = control.pool.acquire().await?;
        if managed_connection(&mut probe, schema_name).await? {
            if access_mode != AccessMode::EnforcedReadOnly {
                bail!("managed CPU-cell schema requires an authorized process binding");
            }
            require_select_only(&mut probe, schema_name).await?;
        }
        drop(probe);
        control.pool.close().await;
        let schema = schema_name.to_owned();
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .acquire_timeout(AUTHORITY_TIMEOUT)
            .after_connect(move |conn, _| {
                let connect_sql = connect_sql.clone();
                let schema = schema.clone();
                Box::pin(async move {
                    tokio::time::timeout(AUTHORITY_TIMEOUT, async {
                        if managed_connection(conn, &schema).await? {
                            if access_mode != AccessMode::EnforcedReadOnly {
                                return Err(sqlx::Error::Protocol("managed CPU-cell schema requires an authorized process binding".into()));
                            }
                            require_select_only(conn, &schema).await?;
                        }
                        conn.execute(connect_sql.as_str()).await?;
                        Ok(())
                    }).await.map_err(|_| sqlx::Error::Protocol("CPU-cell startup guard timed out".into()))?
                })
            }).connect(url).await?;
        Ok(Self {
            pool,
            schema_name: schema_name.to_owned(),
        })
    }

    /// No canonical reads, DDL, or recovery are performed here. Callers use this
    /// small pool only for the role-constrained CPU-cell control functions.
    pub async fn new_control(url: &str, schema_name: &str) -> Result<Self> {
        sanitize_identifier(schema_name)?;
        let pool = PgPoolOptions::new().max_connections(2).acquire_timeout(AUTHORITY_TIMEOUT)
            .after_connect(|conn, _| Box::pin(async move {
                conn.execute("SET statement_timeout = '5s'; SET lock_timeout = '2s'; SET search_path TO pg_catalog;").await?;
                Ok(())
            })).connect(url).await?;
        Ok(Self {
            pool,
            schema_name: schema_name.to_owned(),
        })
    }

    /// Every new canonical connection proves the exact boot binding before it
    /// selects the existing schema. The secret never comes from configuration.
    pub async fn new_owned(
        url: &str,
        schema_name: &str,
        incarnation: &str,
        generation: &str,
        process_id: &str,
        secret: &[u8],
    ) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema_name, AccessMode::ExistingSchema)?;
        let control = Self::new_control(url, schema_name).await?;
        sqlx::query(&format!("SELECT \"{schema_name}\".cpu_cell_assert_owner($1::text::uuid, $2::text::uuid, $3::text::uuid, $4)"))
            .bind(incarnation).bind(generation).bind(process_id).bind(secret)
            .execute(&control.pool).await.wrap_err("authorize CPU-cell startup owner")?;
        control.pool.close().await;
        let assertion = format!("SELECT \"{schema_name}\".cpu_cell_assert_settling_owner($1::text::uuid, $2::text::uuid, $3::text::uuid, $4)");
        let incarnation = incarnation.to_owned();
        let generation = generation.to_owned();
        let process_id = process_id.to_owned();
        let secret = secret.to_vec();
        let pool = PgPoolOptions::new()
            .max_connections(MAX_CONNECTIONS)
            .acquire_timeout(AUTHORITY_TIMEOUT)
            .after_connect(move |conn, _| {
                let assertion = assertion.clone();
                let connect_sql = connect_sql.clone();
                let incarnation = incarnation.clone();
                let generation = generation.clone();
                let process_id = process_id.clone();
                let secret = secret.clone();
                Box::pin(async move {
                    tokio::time::timeout(AUTHORITY_TIMEOUT, async {
                        sqlx::query(&assertion)
                            .bind(incarnation)
                            .bind(generation)
                            .bind(process_id)
                            .bind(secret)
                            .execute(&mut *conn)
                            .await?;
                        conn.execute(connect_sql.as_str()).await?;
                        Ok(())
                    })
                    .await
                    .map_err(|_| {
                        sqlx::Error::Protocol("CPU-cell owner assertion timed out".into())
                    })?
                })
            })
            .connect(url)
            .await?;
        Ok(Self {
            pool,
            schema_name: schema_name.to_owned(),
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
        AUTHORITY_TIMEOUT,
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

    #[test]
    fn schema_names_cannot_be_truncated_into_another_authority() {
        assert!(sanitize_identifier(&"a".repeat(63)).is_ok());
        assert!(sanitize_identifier(&"a".repeat(64)).is_err());
        assert!(sanitize_identifier("").is_err());
    }

    #[tokio::test]
    #[ignore = "requires POP4453_TEST_DATABASE_URL pointing to disposable PostgreSQL"]
    async fn managed_schema_guard_precedes_ddl_and_requires_real_read_only_grants() -> Result<()> {
        let url = std::env::var("POP4453_TEST_DATABASE_URL")?;
        let admin = PgPoolOptions::new()
            .max_connections(2)
            .connect(&url)
            .await?;
        let schema = format!("guard_{}", uuid::Uuid::new_v4().simple());
        let role = format!("reader_{}", uuid::Uuid::new_v4().simple());
        admin.execute(format!("CREATE SCHEMA {schema}; CREATE TABLE {schema}.cpu_cell_authority (id int); CREATE TABLE {schema}.irises (id int); CREATE ROLE {role} LOGIN PASSWORD 'test-only'; GRANT USAGE ON SCHEMA {schema} TO {role}; GRANT SELECT ON {schema}.irises TO {role};").as_str()).await?;
        for mode in [
            AccessMode::ReadWrite,
            AccessMode::ExistingSchema,
            AccessMode::EnforcedReadOnly,
        ] {
            assert!(
                PostgresClient::new(&url, &schema, mode).await.is_err(),
                "privileged/unguarded access must be denied: {mode:?}"
            );
        }
        let mut reader_url = url::Url::parse(&url)?;
        reader_url
            .set_username(&role)
            .map_err(|_| eyre!("test username URL"))?;
        reader_url
            .set_password(Some("test-only"))
            .map_err(|_| eyre!("test password URL"))?;
        let reader =
            PostgresClient::new(reader_url.as_str(), &schema, AccessMode::EnforcedReadOnly).await?;
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM irises")
                .fetch_one(&reader.pool)
                .await?,
            0
        );
        let mut connection = reader.pool.acquire().await?;
        connection
            .execute("SET default_transaction_read_only = off")
            .await?;
        let error = connection
            .execute("INSERT INTO irises VALUES (1)")
            .await
            .unwrap_err();
        assert_eq!(
            error.as_database_error().and_then(|e| e.code()).as_deref(),
            Some("42501")
        );
        drop(connection);
        reader.pool.close().await;
        admin
            .execute(format!("GRANT UPDATE(id) ON {schema}.irises TO {role}").as_str())
            .await?;
        assert!(
            PostgresClient::new(reader_url.as_str(), &schema, AccessMode::EnforcedReadOnly)
                .await
                .is_err()
        );
        let other = format!("legacy_{}", uuid::Uuid::new_v4().simple());
        let legacy = PostgresClient::new(&url, &other, AccessMode::ReadWrite).await?;
        assert!(!is_cpu_cell_managed(&legacy.pool, &other).await?);
        legacy.pool.close().await;
        admin
            .execute(
                format!(
                    "DROP SCHEMA {schema} CASCADE; DROP SCHEMA {other} CASCADE; DROP ROLE {role}"
                )
                .as_str(),
            )
            .await?;
        admin.close().await;
        Ok(())
    }

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
            .execute("CREATE TABLE cpu_cell_authority (id int)")
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
    async fn authority_lookup_failure_is_not_legacy_mode() -> Result<()> {
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
