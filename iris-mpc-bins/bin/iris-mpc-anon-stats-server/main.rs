use std::time::{Duration, Instant};

use clap::Parser;
use eyre::{Context, Result};
use iris_mpc_anon_stats_server::{
    anon_stats::{
        calculate_threshold_a, lift_bundles_1d, test_helper,
        types::{AnonStats1DMapping, AnonStatsOrigin},
    },
    config::{AnonStatsServerConfig, Opt},
    spawn_healthcheck_server,
};
use iris_mpc_common::{
    helpers::{statistics::BucketStatistics, task_monitor::TaskMonitor},
    iris_db::iris::MATCH_THRESHOLD_RATIO,
};
use iris_mpc_common::{job::Eye, tracing::initialize_tracing};
use iris_mpc_cpu::{
    execution::{
        hawk_main::{HawkArgs, Orientation},
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{NetworkSession, Session},
    },
    network::{tcp::build_network_handle, value::NetworkValue},
    protocol::{
        anon_stats::compare_min_threshold_buckets,
        ops::{open_ring, setup_replicated_prf},
    },
};
use itertools::Itertools;
use rand::{rngs::StdRng, Rng, SeedableRng};
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

    tracing::info!("Starting anon stats server.");

    let mut background_tasks = TaskMonitor::new();
    let healthcheck_port = config.healthcheck_port;
    let _healthcheck_handle =
        background_tasks.spawn(async move { spawn_healthcheck_server(healthcheck_port).await });
    background_tasks.check_tasks();
    tracing::info!("Healthcheck server running on port {}", healthcheck_port);

    // set up tcp networking
    let identities = generate_local_identities();
    let role_assignments: RoleAssignment = identities
        .iter()
        .enumerate()
        .map(|(index, id)| (Role::new(index), id.clone()))
        .collect();
    let role_assignments = std::sync::Arc::new(role_assignments);

    // abuse the hawk args struct for now
    let args = HawkArgs {
        party_index: config.party_id,
        addresses: config.addresses.clone(),
        request_parallelism: 1,
        connection_parallelism: 1,
        hnsw_param_M: 0,
        hnsw_param_ef_search: 0,
        hnsw_param_ef_constr: 0,
        disable_persistence: false,
        hnsw_prf_key: None,
        tls: None,
        n_buckets: 0,
        match_distances_buffer_size: 0,
        numa: false,
    };
    let ct = CancellationToken::new();

    // TODO: encapsulate networking setup in a function
    let mut networking = build_network_handle(&args, ct.child_token(), &identities, 8).await?;

    let tcp_sessions = networking
        .as_mut()
        .make_sessions()
        .await
        .context("Making sessions")?;

    let networking_sessions = tcp_sessions
        .into_iter()
        .map(|tcp_session| NetworkSession {
            session_id: tcp_session.id(),
            role_assignments: role_assignments.clone(),
            networking: Box::new(tcp_session),
            own_role: Role::new(config.party_id),
        })
        .collect_vec();
    let mut sessions = Vec::new();
    // todo parallelize session setup
    for mut network_session in networking_sessions {
        let my_session_seed = rand::thread_rng().gen();
        let prf = setup_replicated_prf(&mut network_session, my_session_seed).await?;
        let session = Session {
            network_session,
            prf,
        };
        sessions.push(session);
    }
    tracing::info!("Networking sessions established.");

    let session = &mut sessions[0];
    session
        .network_session
        .send_next(NetworkValue::Bytes(vec![config.party_id as u8]))
        .await?;
    session
        .network_session
        .send_prev(NetworkValue::Bytes(vec![config.party_id as u8]))
        .await?;
    let next_response = session.network_session.receive_next().await?;
    if let NetworkValue::Bytes(b) = next_response {
        if b[0] as usize != ((config.party_id + 1) % 3) {
            tracing::error!("Incorrect prev response value: {:?}", b);
        }
        tracing::info!("Received next response: {:?}", b);
    } else {
        tracing::error!("Unexpected response type for next_response");
    }
    let prev_response = session.network_session.receive_prev().await?;
    if let NetworkValue::Bytes(b) = prev_response {
        if b[0] as usize != ((config.party_id + 2) % 3) {
            tracing::error!("Incorrect prev response value: {:?}", b);
        }
        tracing::info!("Received prev response: {:?}", b);
    } else {
        tracing::error!("Unexpected response type for prev_response");
    }
    tracing::info!("Anon stats server networking test complete.");

    loop {
        tokio::select! {
            // TODO: configure heartbeat interval
            _ = tokio::time::sleep(Duration::from_secs(10)) => {
                tracing::info!("Anon stats server heartbeat - alive and running.");
            }
            _ = shutdown_signal() => {
                tracing::info!("Cancellation token triggered, shutting down anon stats server.");
                break;
            }
        }
        for origin in [
            AnonStatsOrigin {
                side: Eye::Left,
                orientation: Orientation::Normal,
                context: 0,
            },
            // AnonStatsOrigin {
            //     side: Eye::Right,
            //     orientation: Orientation::Normal,
            //     context: 0,
            // },
        ] {
            // load anon stats from DB
            let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);
            let data = test_helper::TestDistances::generate_ground_truth_input(
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

            // TODO: load some dummy data here for testing
            let job_1d = AnonStats1DMapping::new(my_shares);
            let job_size = job_1d.len();
            // sync on the loaded workload with other parties

            let start = Instant::now();
            let job_data = job_1d.into_bundles();
            let lifted_data = lift_bundles_1d(session, &job_data).await?;

            // execute anon stats MPC protocol
            let bucket_result_shares = compare_min_threshold_buckets(
                session,
                translated_thresholds.as_slice(),
                lifted_data.as_slice(),
            )
            .await?;

            let buckets = open_ring(session, &bucket_result_shares).await?;
            let mut anon_stats =
                BucketStatistics::new(job_size, config.n_buckets_1d, config.party_id, origin.side);
            anon_stats.fill_buckets(&buckets, MATCH_THRESHOLD_RATIO, None);

            tracing::info!(
                "Completed anon stats job for origin {:?} of size {} in {:?}. Stats: {:?}",
                origin,
                job_size,
                start.elapsed(),
                anon_stats,
            );
            // verify correctness against ground truth
            assert_eq!(buckets, expected);
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
