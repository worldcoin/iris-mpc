use eyre::{eyre, Result};
use sqlx::{
    migrate::{Migrate, Migrator},
    postgres::PgPoolOptions,
    Executor, PgPool,
};
use std::collections::HashSet;
mod custom;

const MAX_CONNECTIONS: u32 = 100;
const MIGRATION_GRAPH_SWAP_COLUMNS: i64 = 20250512123460;

#[derive(Clone, Debug)]
pub struct PostgresClient {
    pub pool: PgPool,
    pub schema_name: String,
    pub access_mode: AccessMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ReadOnly,
    ReadWrite,
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

fn sql_switch_schema(schema_name: &str, access_mode: AccessMode) -> Result<String> {
    sanitize_identifier(schema_name)?;

    if access_mode == AccessMode::ReadOnly {
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
            access_mode,
            schema_name: schema_name.to_string(),
        })
    }

    pub async fn migrate(&self) {
        tracing::info!("Running migrations...");

        if self.access_mode == AccessMode::ReadOnly {
            tracing::info!("Not migrating client in read-only mode");
            return;
        }

        let migrator: Migrator = sqlx::migrate!("./../migrations");
        let mut conn = self
            .pool
            .acquire()
            .await
            .expect("Failed to acquire connection");
        conn.lock()
            .await
            .expect("failed to lock the migrations database");
        conn.ensure_migrations_table()
            .await
            .expect("failed to ensure migrations table");

        let applied_migrations = conn
            .list_applied_migrations()
            .await
            .expect("failed to get applied migrations");
        let applied_migrations: HashSet<i64> =
            applied_migrations.into_iter().map(|m| m.version).collect();

        for migration in migrator.iter() {
            let version = migration.version;
            let description = &migration.description;
            if applied_migrations.contains(&version) {
                tracing::info!("Already applied migration {}: {}", version, description);
                continue;
            }
            match version {
                MIGRATION_GRAPH_SWAP_COLUMNS => {
                    tracing::info!(
                        "Running converter for migration {}: {}",
                        version,
                        description
                    );
                    custom::graph::links_to_bytea(&self.pool)
                        .await
                        .expect("failed to convert graph links");
                    custom::graph::entry_points_to_bytea(&self.pool)
                        .await
                        .expect("failed to convert graph entry points");
                }
                _ => {}
            }
            tracing::info!("Applying migration {}: {}", version, description);
            conn.apply(&migration)
                .await
                .expect("failed to apply migration");

            if final_migration.map(|x| x == version).unwrap_or_default() {
                tracing::info!("terminating migrations early due to user's request");
                break;
            }
        }
        conn.unlock()
            .await
            .expect("failed to release lock on migrations table");
    }
}
