use std::io::Read;
use std::ops::Range;

use clap::Parser;
use eyre::Result;
use futures::TryStreamExt;
use iris_mpc_common::galois::degree4::basis::Monomial;
use iris_mpc_common::galois::degree4::GaloisRingElement;
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::{DbStoredIris, Store, StoredIrisRef};
use iris_mpc_upgrade::{
    config::ReRandomizeDbConfig,
    utils::{install_tracing, spawn_healthcheck_server},
};
use tokio::task::JoinSet;
use tracing::Level;

const APP_NAME: &str = "SMPC";

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let config = ReRandomizeDbConfig::parse();

    if config.master_seed.len() <= 32 {
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

    tracing::info!(
        "Healthcheck server running on port {}.",
        config.healthcheck_port
    );

    let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
    let postgres_client =
        PostgresClient::new(&config.db_url, &schema_name, AccessMode::ReadWrite).await?;
    let store = Store::new(&postgres_client).await?;

    rerandomize_db(&store, config).await?;

    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn rerandomize_db(store: &Store, config: ReRandomizeDbConfig) -> Result<()> {
    tracing::info!("Rerandomizing database for party ID: {}", config.party_id);

    let max_id = store.get_max_serial_id().await?;

    let chunk_len = max_id.div_ceil(config.num_tasks);

    let mut tasks = JoinSet::new();

    for i in 0..config.num_tasks {
        let start = 1 + i * chunk_len;
        let end = std::cmp::min(start + chunk_len, max_id + 1);
        let store = store.clone();
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
                let chunk: Result<Vec<DbStoredIris>, _> = store
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
                    .map(|iris| randomize_iris(iris, &master_seed, party_id as usize))
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

                let mut tx = match store.tx().await {
                    Ok(tx) => tx,
                    Err(e) => {
                        tracing::error!("Failed to start transaction: {e}");
                        continue;
                    }
                };

                match store
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

fn randomize_iris(
    iris: DbStoredIris,
    master_seed: &str,
    party_id: usize,
) -> (
    i64,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(master_seed.as_bytes());
    hasher.update(&iris.id().to_le_bytes());
    let mut xof = hasher.finalize_xof();

    let (mut left_code, mut left_mask, mut right_code, mut right_mask) = (
        GaloisRingIrisCodeShare {
            id: party_id as usize + 1,
            coefs: iris.left_code().try_into().unwrap(),
        },
        GaloisRingTrimmedMaskCodeShare {
            id: party_id as usize + 1,
            coefs: iris.left_mask().try_into().unwrap(),
        },
        GaloisRingIrisCodeShare {
            id: party_id as usize + 1,
            coefs: iris.right_code().try_into().unwrap(),
        },
        GaloisRingTrimmedMaskCodeShare {
            id: party_id as usize + 1,
            coefs: iris.right_mask().try_into().unwrap(),
        },
    );

    randomize_galois_ring_coefs(&mut left_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut left_mask.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_mask.coefs, &mut xof, party_id);
    (iris.id(), left_code, left_mask, right_code, right_mask)
}

fn randomize_galois_ring_coefs(coefs: &mut [u16], xof: &mut blake3::OutputReader, party_id: usize) {
    for coefs in coefs.chunks_mut(4) {
        assert!(coefs.len() == 4, "Expected 4 coefficients per chunk");
        let mut gr = GaloisRingElement::<Monomial>::from_coefs(coefs.try_into().unwrap());
        let mut r = [0u16; 4];
        xof.read(bytemuck::cast_slice_mut(&mut r[..]))
            .expect("can read from xof");
        let mut r = GaloisRingElement::<Monomial>::from_coefs(r);
        r = r * GaloisRingElement::<Monomial>::EXCEPTIONAL_SEQUENCE[party_id];
        gr = gr + r;
        coefs.copy_from_slice(&gr.coefs[..]);
    }
}
