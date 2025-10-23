use std::time::Duration;

use clap::Parser;
use eyre::{Context, Result};
use iris_mpc_anon_stats_server::{
    config::AnonStatsServerConfig, config::Opt, spawn_healthcheck_server,
};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_cpu::{
    execution::{
        hawk_main::HawkArgs,
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{NetworkSession, Session},
    },
    network::{tcp::build_network_handle, value::NetworkValue},
    protocol::ops::setup_replicated_prf,
};
use itertools::Itertools;
use rand::Rng;
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: AnonStatsServerConfig = AnonStatsServerConfig::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

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
    let mut networking = build_network_handle(&args, ct, &identities, 8).await?;

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
    tokio::time::sleep(Duration::from_secs(5)).await;
    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}
