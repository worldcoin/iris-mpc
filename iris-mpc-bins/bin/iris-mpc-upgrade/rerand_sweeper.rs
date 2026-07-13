use clap::Parser;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade::config::RerandSweeperConfig;
use iris_mpc_upgrade::rerand_v2::sweeper::run_single_pass;
use iris_mpc_upgrade::utils::install_tracing;

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let config = RerandSweeperConfig::parse();
    let postgres =
        PostgresClient::new(&config.db_url, &config.schema_name, AccessMode::ReadOnly).await?;
    let store = Store::new(&postgres).await?;
    let aws = aws_config::from_env().load().await;
    let outcome = run_single_pass(
        &store,
        &aws_sdk_s3::Client::new(&aws),
        &aws_sdk_secretsmanager::Client::new(&aws),
        &config,
    )
    .await?;
    tracing::info!(
        epoch = outcome.epoch,
        retargeted = outcome.retargeted,
        semantic_write_races = outcome.semantic_write_races,
        "rerandomization pass complete"
    );
    Ok(())
}
