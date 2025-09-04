use std::ops::Range;

use clap::Parser;
use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{DbStoredIris, Store, StoredIrisRef};
use iris_mpc_upgrade::rerandomization::randomize_iris;
use iris_mpc_upgrade::{
    config::ReRandomizeDbConfig,
    utils::{install_tracing, spawn_healthcheck_server},
};
use tokio::task::JoinSet;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let config = ReRandomizeDbConfig::parse();

    if config.master_seed.len() < 32 {
        eyre::bail!("Master seed must be at least 32 characters.");
    }

    tracing::info!("Starting healthcheck server.");

    let mut background_tasks = TaskMonitor::new();
    let _health_check_abort = background_tasks
        .spawn(async move { spawn_healthcheck_server(config.healthcheck_port).await });
    background_tasks.check_tasks();
    tracing::info!(
        "Healthcheck server running on port {}.",
        config.healthcheck_port.clone()
    );

    let postgres_client_read = PostgresClient::new(
        &config.source_db_url,
        &config.source_schema_name,
        AccessMode::ReadOnly,
    )
    .await?;
    let postgres_client_write = PostgresClient::new(
        &config.dest_db_url,
        &config.dest_schema_name,
        AccessMode::ReadWrite,
    )
    .await?;
    let read_store = Store::new(&postgres_client_read).await?;
    let write_store = Store::new(&postgres_client_write).await?;

    rerandomize_db(&read_store, &write_store, config).await?;

    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn rerandomize_db(
    read_store: &Store,
    write_store: &Store,
    config: ReRandomizeDbConfig,
) -> Result<()> {
    tracing::info!("Rerandomizing database for party ID: {}", config.party_id);

    let max_id = read_store.get_max_serial_id().await?;

    let chunk_len = max_id.div_ceil(config.num_tasks);

    let mut tasks = JoinSet::new();

    for i in 0..config.num_tasks {
        let start = 1 + i * chunk_len;
        let end = std::cmp::min(start + chunk_len, max_id + 1);
        let read_store = read_store.clone();
        let write_store = write_store.clone();
        let party_id = config.party_id;
        let master_seed = config.master_seed.clone();

        tasks.spawn(async move {
            for chunk_start in (start..end).step_by(config.chunk_size) {
                let chunk_end = std::cmp::min(chunk_start + config.chunk_size, end);
                let span = tracing::span!(
                    Level::INFO,
                    "Processing chunk {} to {} for party ID: {}",
                    chunk_start,
                    chunk_end,
                    party_id
                );
                let _span = span.enter();
                let chunk: Result<Vec<DbStoredIris>, _> = read_store
                    .stream_irises_in_range(Range {
                        start: chunk_start as u64,
                        end: chunk_end as u64,
                    })
                    .try_collect()
                    .await;
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Failed to fetch chunk {chunk_start}-{chunk_end}: {e}");
                        continue;
                    }
                };

                tracing::info!(
                    "Fetched chunk {} to {} for party ID: {}",
                    chunk_start,
                    chunk_end,
                    party_id
                );

                let rerandomized_chunk: Vec<_> = chunk
                    .into_iter()
                    .map(|iris| randomize_iris(iris, master_seed.as_bytes(), party_id as usize))
                    .collect();
                tracing::info!(
                    "Rerandomized chunk  {} to {} for party ID: {}",
                    chunk_start,
                    chunk_end,
                    party_id
                );

                let inserted_chunk: Vec<_> = rerandomized_chunk
                    .iter()
                    .map(
                        |(id, left_code, left_mask, right_code, right_mask)| StoredIrisRef {
                            id: *id,
                            left_code: &left_code.coefs,
                            left_mask: &left_mask.coefs,
                            right_code: &right_code.coefs,
                            right_mask: &right_mask.coefs,
                        },
                    )
                    .collect();

                let mut tx = match write_store.tx().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        tracing::error!("Failed to start transaction: {e}");
                        continue;
                    }
                };

                match write_store
                    .insert_irises_overriding(&mut tx, &inserted_chunk)
                    .await
                {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("Failed to insert rerandomized iris chunk: {e}");
                        continue;
                    }
                }

                if let Err(e) = tx.commit().await {
                    tracing::error!("Failed to commit transaction: {e}");
                } else {
                    tracing::info!(
                        "Successfully committed rerandomized chunk {} to {} for party ID: {}",
                        chunk_start,
                        chunk_end,
                        party_id
                    );
                }
            }
        });
    }

    tasks.join_all().await;
    tracing::info!("All rerandomization tasks completed.");

    Ok(())
}
