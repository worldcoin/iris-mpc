use std::path::Path;

use eyre::{Context, Result};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use sqlx::migrate::Migrator;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL")?;
    let schema_name = std::env::var("SCHEMA_NAME")?;
    // The standalone legacy migrator must not bypass managed startup guards.
    let postgres = PostgresClient::new(&database_url, &schema_name, AccessMode::ExistingSchema)
        .await
        .with_context(|| "Could not authorize PostgreSQL migration access")?;
    let migrator = Migrator::new(Path::new("./migrations"))
        .await
        .with_context(|| "Could not create Migrator!")?;
    migrator
        .run(&postgres.pool)
        .await
        .with_context(|| "Could not run migration!")?;
    Ok(())
}
