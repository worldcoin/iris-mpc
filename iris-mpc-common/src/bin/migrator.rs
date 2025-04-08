use std::path::Path;

use eyre::{Context, Result};
use sqlx::{migrate::Migrator, postgres::PgPoolOptions, Executor};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").unwrap();
    let postgres_pool = PgPoolOptions::new()
        .max_connections(1)
        .after_connect(|conn, _meta| {
            let schema_name = std::env::var("SCHEMA_NAME").unwrap();
            let query = format!("SET search_path = '{}';", schema_name.clone());
            Box::pin(async move {
                conn.execute(query.as_str()).await?;
                Ok(())
            })
        })
        .connect(&database_url)
        .await
        .with_context(|| "Could not connect to PostgreSQL!")?;

    let migrator = Migrator::new(Path::new("./migrations"))
        .await
        .with_context(|| "Could not create Migrator!")?;
    migrator
        .run(&postgres_pool)
        .await
        .with_context(|| "Could not run migration!")?;
    Ok(())
}
