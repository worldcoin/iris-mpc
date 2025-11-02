use std::time::{Duration, Instant};

use clap::Parser;
use eyre::{Context, Result};
use iris_mpc_anon_stats_server::{
    anon_stats::{
        self, calculate_threshold_a,
        store::AnonStatsStore,
        test_helper_1d, test_helper_2d,
        types::{AnonStatsContext, AnonStatsMapping, AnonStatsOrigin},
    },
    config::{AnonStatsServerConfig, Opt},
    spawn_healthcheck_server,
};
use iris_mpc_common::{
    helpers::{statistics::BucketStatistics2D, task_monitor::TaskMonitor},
    iris_db::iris::MATCH_THRESHOLD_RATIO,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_common::{job::Eye, tracing::initialize_tracing};
use iris_mpc_cpu::{
    execution::hawk_main::Orientation,
    network::tcp::{build_network_handle, NetworkHandleArgs},
};
use itertools::Itertools;
use rand::{rngs::StdRng, SeedableRng};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: AnonStatsServerConfig = AnonStatsServerConfig::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());
    config.apply_party_network_defaults()?;

    let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    tracing::info!("Connecting to database");
    let postgres_client = PostgresClient::new(
        &config.db_url,
        &config.db_schema_name,
        AccessMode::ReadWrite,
    )
    .await?;
    let anon_stats_store = AnonStatsStore::new(&postgres_client).await?;

    tracing::info!("Starting anon stats server.");

    let mut background_tasks = TaskMonitor::new();
    let healthcheck_port = config.healthcheck_port;
    let _healthcheck_handle =
        background_tasks.spawn(async move { spawn_healthcheck_server(healthcheck_port).await });
    background_tasks.check_tasks();
    tracing::info!("Healthcheck server running on port {}", healthcheck_port);

    // set up tcp networking
    let args = NetworkHandleArgs {
        party_index: config.party_id,
        addresses: config.addresses.clone(),
        connection_parallelism: 8,
        request_parallelism: 8,
        sessions_per_request: 1,
        tls: None,
    };
    let ct = CancellationToken::new();

    let mut networking = build_network_handle(args, ct.child_token()).await?;
    let mut sessions = networking
        .as_mut()
        .make_sessions()
        .await
        .context("Making sessions")?;
    tracing::info!("Networking sessions established.");

    let session = &mut sessions[0];
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    loop {
        tokio::select! {
            // TODO: configure heartbeat interval
            _ = interval.tick() => {
                tracing::info!("Anon stats server heartbeat - alive and running.");
            }
            _ = shutdown_signal() => {
                tracing::info!("Cancellation token triggered, shutting down anon stats server.");
                break;
            }
        }
        for origin in [
            // 1d anon stats jobs for left and right eyes, GPU and HSNW contexts
            AnonStatsOrigin {
                side: Some(Eye::Left),
                orientation: Orientation::Normal,
                context: AnonStatsContext::GPU,
            },
            AnonStatsOrigin {
                side: Some(Eye::Right),
                orientation: Orientation::Normal,
                context: AnonStatsContext::GPU,
            },
            AnonStatsOrigin {
                side: Some(Eye::Left),
                orientation: Orientation::Normal,
                context: AnonStatsContext::HNSW,
            },
            AnonStatsOrigin {
                side: Some(Eye::Right),
                orientation: Orientation::Normal,
                context: AnonStatsContext::HNSW,
            },
        ] {
            tracing::info!("Starting anon stats job for origin {:?}", origin);

            // prepare test data
            let (expected, my_shares) = {
                let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);
                let data = test_helper_1d::TestDistances::generate_ground_truth_input(
                    &mut StdRng::seed_from_u64(123),
                    1000,
                    6,
                );
                let expected = data.ground_truth_buckets(&translated_thresholds);
                let my_shares = match config.party_id {
                    0 => data.shares0,
                    1 => data.shares1,
                    2 => data.shares2,
                    _ => panic!("Invalid party index"),
                }
                .into_iter()
                .enumerate()
                .map(|(i, share)| (i as i64, share))
                .collect_vec();
                (expected, my_shares)
            };

            // need to call different methods because GPU and HSNW anon stats inputs are different formats
            // GPU is u16, while HSNW is lifted u32
            let anon_stats = match origin.context {
                AnonStatsContext::GPU => {
                    tracing::info!("Inserting test anon stats data into DB");
                    anon_stats_store
                        .insert_anon_stats_batch_1d(&my_shares, origin)
                        .await?;

                    // we get the size of available anon stats from all parties
                    tracing::info!("Loading anon stats job data from DB");
                    let available_anon_stats = match usize::try_from(
                        anon_stats_store.num_available_anon_stats_1d(origin).await?,
                    ) {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::error!("Failed to convert num_available_anon_stats to usize: {:?}, using 0 as a standing", e);
                            0
                        }
                    };
                    let min_job_size = iris_mpc_anon_stats_server::sync::sync_on_job_sizes(
                        session,
                        available_anon_stats,
                    )
                    .await?;
                    if min_job_size < config.min_1d_job_size {
                        tracing::info!("Not enough available anon stats for origin {:?} (available: {}, minimum required: {}), will retry later", origin, min_job_size, config.min_1d_job_size);
                        continue;
                    }

                    tracing::info!(
                        "Loading {min_job_size} available anon stats for origin {:?} from DB",
                        origin,
                    );
                    let (ids, my_anon_stats_shares) = anon_stats_store
                        .get_available_anon_stats_1d(origin, min_job_size)
                        .await?;

                    let job_1d = AnonStatsMapping::new(my_anon_stats_shares);
                    let job_hash = job_1d.get_id_hash();

                    let hashes_match =
                        iris_mpc_anon_stats_server::sync::sync_on_id_hash(session, job_hash)
                            .await?;
                    if !hashes_match {
                        tracing::info!("Mismatched job data detected among parties, will retry at a later point");
                    }

                    let start = Instant::now();
                    let anon_stats =
                        anon_stats::process_1d_anon_stats_job(session, job_1d, &origin, &config)
                            .await?;

                    tracing::info!(
                        "Completed anon stats job for origin {:?} of size {} in {:?}. Stats: {:?}",
                        origin,
                        min_job_size,
                        start.elapsed(),
                        anon_stats,
                    );
                    tracing::info!("Anon stats calculation complete, marking as processed in DB");
                    anon_stats_store.mark_anon_stats_processed_1d(&ids).await?;
                    anon_stats
                }
                AnonStatsContext::HNSW => {
                    tracing::info!("Inserting test anon stats data into DB");
                    let my_shares = my_shares.into_iter().map(|(_, share)| share).collect_vec();
                    let lifted = anon_stats::lift_bundles_1d(session, &my_shares).await?;
                    let my_shares = lifted
                        .into_iter()
                        .enumerate()
                        .map(|(i, share)| (i as i64, share))
                        .collect_vec();
                    anon_stats_store
                        .insert_anon_stats_batch_1d_lifted(&my_shares, origin)
                        .await?;

                    // we get the size of available anon stats from all parties
                    tracing::info!("Loading anon stats job data from DB");
                    let available_anon_stats = match usize::try_from(
                        anon_stats_store
                            .num_available_anon_stats_1d_lifted(origin)
                            .await?,
                    ) {
                        Ok(n) => n,
                        Err(e) => {
                            tracing::error!("Failed to convert num_available_anon_stats to usize: {:?}, using 0 as a standing", e);
                            0
                        }
                    };
                    let min_job_size = iris_mpc_anon_stats_server::sync::sync_on_job_sizes(
                        session,
                        available_anon_stats,
                    )
                    .await?;
                    if min_job_size < config.min_1d_job_size {
                        tracing::info!("Not enough available anon stats for origin {:?} (available: {}, minimum required: {}), will retry later", origin, min_job_size, config.min_1d_job_size);
                        continue;
                    }

                    tracing::info!(
                        "Loading {min_job_size} available anon stats for origin {:?} from DB",
                        origin,
                    );
                    let (ids, my_anon_stats_shares) = anon_stats_store
                        .get_available_anon_stats_1d_lifted(origin, min_job_size)
                        .await?;

                    let job_1d = AnonStatsMapping::new(my_anon_stats_shares);
                    let job_hash = job_1d.get_id_hash();

                    let hashes_match =
                        iris_mpc_anon_stats_server::sync::sync_on_id_hash(session, job_hash)
                            .await?;
                    if !hashes_match {
                        tracing::info!("Mismatched job data detected among parties, will retry at a later point");
                    }

                    let start = Instant::now();
                    let anon_stats = anon_stats::process_1d_lifted_anon_stats_job(
                        session, job_1d, &origin, &config,
                    )
                    .await?;

                    tracing::info!(
                        "Completed anon stats job for origin {:?} of size {} in {:?}. Stats: {:?}",
                        origin,
                        min_job_size,
                        start.elapsed(),
                        anon_stats,
                    );
                    tracing::info!("Anon stats calculation complete, marking as processed in DB");
                    anon_stats_store
                        .mark_anon_stats_processed_1d_lifted(&ids)
                        .await?;
                    anon_stats
                }
            };

            let buckets = anon_stats
                .buckets
                .iter()
                .fold(
                    (0u32, Vec::with_capacity(config.n_buckets_1d)),
                    |(total, mut vec), b| {
                        let cummulative = total + b.count as u32;

                        vec.push(cummulative);
                        (cummulative, vec)
                    },
                )
                .1;
            // verify correctness against ground truth
            assert_eq!(buckets, expected);
        }

        // 2d anon stats
        #[allow(clippy::single_element_loop)]
        for origin in [
            AnonStatsOrigin {
                side: None,
                orientation: Orientation::Normal,
                context: AnonStatsContext::GPU,
            },
            // TODO HNSW 2D stats
        ] {
            tracing::info!("Starting 2D anon stats job for origin {:?}", origin);
            // prepare test data
            let (expected, my_shares) = {
                let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);
                let data = test_helper_2d::TestDistances::generate_ground_truth_input(
                    &mut StdRng::seed_from_u64(123),
                    1000,
                    6,
                );
                let expected = data.ground_truth_buckets(&translated_thresholds);
                let my_shares = match config.party_id {
                    0 => data.shares0,
                    1 => data.shares1,
                    2 => data.shares2,
                    _ => panic!("Invalid party index"),
                }
                .into_iter()
                .enumerate()
                .map(|(i, share)| (i as i64, share))
                .collect_vec();
                (expected, my_shares)
            };

            tracing::info!("Inserting 2D test anon stats data into DB");
            anon_stats_store
                .insert_anon_stats_batch_2d(&my_shares, origin)
                .await?;

            // we get the size of available anon stats from all parties
            tracing::info!("Loading anon stats job data from DB");
            let available_anon_stats = match usize::try_from(
                anon_stats_store.num_available_anon_stats_2d(origin).await?,
            ) {
                Ok(n) => n,
                Err(e) => {
                    tracing::error!("Failed to convert num_available_anon_stats to usize: {:?}, using 0 as a standing", e);
                    0
                }
            };
            let min_job_size =
                iris_mpc_anon_stats_server::sync::sync_on_job_sizes(session, available_anon_stats)
                    .await?;
            if min_job_size < config.min_1d_job_size {
                tracing::info!("Not enough available anon stats for origin {:?} (available: {}, minimum required: {}), will retry later", origin, min_job_size, config.min_1d_job_size);
                continue;
            }

            tracing::info!(
                "Loading {min_job_size} available anon stats for origin {:?} from DB",
                origin,
            );
            let (ids, my_anon_stats_shares) = anon_stats_store
                .get_available_anon_stats_2d(origin, min_job_size)
                .await?;

            let job_2d = AnonStatsMapping::new(my_anon_stats_shares);
            let job_hash = job_2d.get_id_hash();

            let hashes_match =
                iris_mpc_anon_stats_server::sync::sync_on_id_hash(session, job_hash).await?;
            if !hashes_match {
                tracing::info!(
                    "Mismatched job data detected among parties, will retry at a later point"
                );
            }

            let start = Instant::now();
            let anon_stats =
                anon_stats::process_2d_anon_stats_job(session, job_2d, &config).await?;

            tracing::info!(
                "Completed anon stats job for origin {:?} of size {} in {:?}. Stats: {:?}",
                origin,
                min_job_size,
                start.elapsed(),
                anon_stats,
            );
            tracing::info!("Anon stats calculation complete, marking as processed in DB");
            anon_stats_store.mark_anon_stats_processed_2d(&ids).await?;

            let mut expected_stats =
                BucketStatistics2D::new(min_job_size, config.n_buckets_1d, config.party_id); // verify correctness against ground truth
            expected_stats.fill_buckets(&expected, MATCH_THRESHOLD_RATIO, None);
            assert_eq!(anon_stats.buckets, expected_stats.buckets);
        }
    }

    // TODO: how to wait for networking shutdown?
    ct.cancel();
    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("Failed to install SIGINT handler");
        let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler");

        tokio::select! {
            _ = sigint.recv() => {},
            _ = sigterm.recv() => {},
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl-C handler");
    }

    tracing::info!("Shutdown signal received.");
}
