use eyre::{eyre, Result};
use sqlx::{postgres::PgPoolOptions, Executor, PgPool};

const MAX_CONNECTIONS: u32 = 100;

#[derive(Clone, Debug)]
pub struct PostgresClient {
    pub pool: PgPool,
    pub schema_name: String,
    pub access_mode: AccessMode,
}

// Helper type: name of a PostgreSQL schema.
pub type PostgresSchemaName = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    ReadOnly,
    ReadWrite,
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input
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

        sqlx::migrate!("./../migrations")
            .run(&self.pool)
            .await
            .expect("Failed to run migrations");
    }
}
