//! Deployment-side handoff installation and intent. Never enables serving.
use eyre::{bail, Result};
use iris_mpc_common::postgres::PostgresClient;
use iris_mpc_store::cpu_cell::CpuCellHandoffs;
use sqlx::Executor;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let values: Vec<&str> = args.iter().map(String::as_str).collect();
    let url = std::env::var("CPU_CELL_DATABASE_URL")?;
    let Some(schema) = values.get(1) else {
        bail!("expected install SCHEMA, request SCHEMA ID SOURCE DESTINATION, or status SCHEMA ID");
    };
    let client = PostgresClient::new_control(&url, schema).await?;
    let handoffs = CpuCellHandoffs::new(client.pool.clone(), schema)?;
    match values.as_slice() {
        ["install", _] => {
            let mut tx = client.pool.begin().await?;
            tx.execute(format!("SET LOCAL search_path TO \"{schema}\"").as_str())
                .await?;
            tx.execute(include_str!("../../scripts/cpu-cell-ownership.sql"))
                .await?;
            tx.commit().await?;
        }
        ["request", _, id, source, destination] => {
            handoffs
                .request(
                    Uuid::parse_str(id)?,
                    Uuid::parse_str(source)?,
                    Uuid::parse_str(destination)?,
                )
                .await?;
        }
        ["status", _, id] => println!("{:#?}", handoffs.get(Uuid::parse_str(id)?).await?),
        _ => bail!(
            "expected install SCHEMA, request SCHEMA ID SOURCE DESTINATION, or status SCHEMA ID"
        ),
    }
    client.pool.close().await;
    Ok(())
}
