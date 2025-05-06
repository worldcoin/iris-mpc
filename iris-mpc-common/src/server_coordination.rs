use crate::config::Config;
use crate::helpers::fetch_index::fetch_height_of_indexed;
use crate::helpers::shutdown_handler::ShutdownHandler;
use crate::helpers::sync::{SyncResult, SyncState};
use crate::helpers::task_monitor::TaskMonitor;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{bail, eyre, Error, Result, WrapErr};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReadyProbeResponse {
    pub image_name: String,
    pub uuid: String,
    pub shutting_down: bool,
}

// Returns a new task monitor.
pub fn init_task_monitor() -> TaskMonitor {
    tracing::info!("Preparing task monitor");
    TaskMonitor::new()
}

pub fn get_check_addresses(
    hostnames: Vec<String>,
    ports: Vec<String>,
    endpoint: &str,
) -> Vec<String> {
    hostnames
        .iter()
        .zip(ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host, port, endpoint))
        .collect::<Vec<String>>()
}

/// Initializes and starts HTTP server for coordinating healthcheck, readiness,
/// and synchronization between MPC nodes.
///
/// Note: returns a reference to a readiness flag, an `AtomicBool`, which can later
/// be set to indicate to other MPC nodes that this server is ready for operation.
pub async fn start_coordination_server(
    config: &Config,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    my_state: &SyncState,
) -> Arc<AtomicBool> {
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let is_ready_flag = Arc::new(AtomicBool::new(false));

    let health_shutdown_handler = Arc::clone(shutdown_handler);
    let health_check_port = config.hawk_server_healthcheck_port;

    let _health_check_abort = task_monitor.spawn({
        let uuid = uuid::Uuid::new_v4().to_string();
        let is_ready_flag = Arc::clone(&is_ready_flag);
        let ready_probe_response = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: false,
            uuid: uuid.clone(),
        };
        let ready_probe_response_shutdown = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: true,
            uuid: uuid.clone(),
        };
        let serialized_response = serde_json::to_string(&ready_probe_response)
            .expect("Serialization to JSON to probe response failed");
        let serialized_response_shutdown = serde_json::to_string(&ready_probe_response_shutdown)
            .expect("Serialization to JSON to probe response failed");
        tracing::info!("Healthcheck probe response: {}", serialized_response);
        let my_state = my_state.clone();
        async move {
            // Generate a random UUID for each run.
            let app = Router::new()
                .route(
                    "/health",
                    get(move || {
                        let shutdown_handler_clone = Arc::clone(&health_shutdown_handler);
                        async move {
                            if shutdown_handler_clone.is_shutting_down() {
                                serialized_response_shutdown.clone()
                            } else {
                                serialized_response.clone()
                            }
                        }
                    }),
                )
                .route(
                    "/ready",
                    get({
                        // We are only ready once this flag is set to true.
                        let is_ready_flag = Arc::clone(&is_ready_flag);
                        move || async move {
                            if is_ready_flag.load(Ordering::SeqCst) {
                                "ready".into_response()
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE.into_response()
                            }
                        }
                    }),
                )
                .route(
                    "/startup-sync",
                    get(move || async move { serde_json::to_string(&my_state).unwrap() }),
                )
                .route(
                    "/height-of-graph-genesis-indexation",
                    get({
                        let height = fetch_height_of_indexed().await;
                        let is_ready_flag = Arc::clone(&is_ready_flag);
                        move || async move {
                            if is_ready_flag.load(Ordering::SeqCst) {
                                height.to_string().into_response()
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE.into_response()
                            }
                        }
                    }),
                );
            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", health_check_port))
                .await
                .wrap_err("healthcheck listener bind error")?;
            axum::serve(listener, app)
                .await
                .wrap_err("healthcheck listener server launch error")?;

            Ok::<(), Error>(())
        }
    });

    tracing::info!(
        "Healthcheck and Readiness server running on port {}.",
        health_check_port.clone()
    );

    is_ready_flag
}

/// Awaits until other MPC nodes respond to "ready" queries
/// indicating that their coordination servers are running.
///
/// Note: The response to this query is expected initially to be `503 Service Unavailable`.
pub async fn wait_for_others_unready(config: &Config) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let party_id = config.party_id;

    let unready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(party_id + 2) % 3];
        let mut connected_but_unready = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;

                if res.is_ok() && res.unwrap().status() == StatusCode::SERVICE_UNAVAILABLE {
                    connected_but_unready[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected_but_unready.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be unready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        unready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be unready.");
            return Err(eyre!("Timeout waiting for all nodes to be unready."));
        }
    };
    tracing::info!("All nodes are starting up.");

    Ok(())
}

/// Starts a heartbeat task which periodically polls the "health" endpoints of
/// all other MPC nodes to ensure that the other nodes are still running and
/// responding to network requests.
pub async fn init_heartbeat_task(
    config: &Config,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);

    let all_health_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "health",
    );

    let party_id = config.party_id;
    let image_name = config.image_name.clone();
    let heartbeat_initial_retries = config.heartbeat_initial_retries;
    let heartbeat_interval_secs = config.heartbeat_interval_secs;

    let heartbeat_shutdown_handler = Arc::clone(shutdown_handler);
    let _heartbeat = task_monitor.spawn(async move {
        let next_node = &all_health_addresses[(party_id + 1) % 3];
        let prev_node = &all_health_addresses[(party_id + 2) % 3];
        let mut last_response = [String::default(), String::default()];
        let mut connected = [false, false];
        let mut retries = [0, 0];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;
                if res.is_err() || !res.as_ref().unwrap().status().is_success() {
                    // If it's the first time after startup, we allow a few retries to let the other
                    // nodes start up as well.
                    if last_response[i] == String::default()
                        && retries[i] < heartbeat_initial_retries
                    {
                        retries[i] += 1;
                        tracing::warn!("Node {} did not respond with success, retrying...", host);
                        continue;
                    }
                    tracing::info!(
                        "Node {} did not respond with success, starting graceful shutdown",
                        host
                    );
                    // if the nodes are still starting up and they get a failure - we can panic and
                    // not start graceful shutdown
                    if last_response[i] == String::default() {
                        panic!(
                            "Node {} did not respond with success during heartbeat init phase, \
                             killing server...",
                            host
                        );
                    }

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has not completed health check, therefore graceful shutdown \
                             has been triggered",
                            host
                        );
                    } else {
                        tracing::info!("Node {} has already started graceful shutdown.", host);
                    }
                    continue;
                }

                let probe_response = res
                    .unwrap()
                    .json::<ReadyProbeResponse>()
                    .await
                    .expect("Deserialization of probe response failed");
                if probe_response.image_name != image_name {
                    // Do not create a panic as we still can continue to process before its
                    // updated
                    tracing::error!(
                        "Host {} is using image {} which differs from current node image: {}",
                        host,
                        probe_response.image_name.clone(),
                        image_name
                    );
                }
                if last_response[i] == String::default() {
                    last_response[i] = probe_response.uuid;
                    connected[i] = true;

                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        if let Some(tx) = heartbeat_tx.take() {
                            tx.send(()).unwrap();
                        }
                    }
                } else if probe_response.uuid != last_response[i] {
                    // If the UUID response is different, the node has restarted without us
                    // noticing. Our main NCCL connections cannot recover from
                    // this, so we panic.
                    panic!("Node {} seems to have restarted, killing server...", host);
                } else if probe_response.shutting_down {
                    tracing::info!("Node {} has starting graceful shutdown", host);

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has starting graceful shutdown, therefore triggering \
                             graceful shutdown",
                            host
                        );
                    }
                } else {
                    tracing::info!("Heartbeat: Node {} is healthy", host);
                }
            }

            tokio::time::sleep(Duration::from_secs(heartbeat_interval_secs)).await;
        }
    });

    tracing::info!("Heartbeat starting...");
    heartbeat_rx.await?;
    tracing::info!("Heartbeat on all nodes started.");

    Ok(())
}

/// Retrieves synchronization state of other MPC nodes.  This data is
/// used to ensure that all nodes are in a consistent state prior
/// to starting MPC operations.
pub async fn get_others_sync_state(config: &Config, my_state: &SyncState) -> Result<SyncResult> {
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");

    let all_startup_sync_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "startup-sync",
    );

    let next_node = &all_startup_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_startup_sync_addresses[(config.party_id + 2) % 3];

    tracing::info!("Database store length is: {}", my_state.db_len);
    let mut states = vec![my_state.clone()];
    for host in [next_node, prev_node].iter() {
        let res = reqwest::get(host.as_str()).await;
        match res {
            Ok(res) => {
                let state: SyncState = match res.json().await {
                    Ok(state) => state,
                    Err(e) => {
                        tracing::error!("Failed to parse sync state from party {}: {:?}", host, e);
                        panic!(
                            "could not get sync state from party {}, trying to restart",
                            host
                        );
                    }
                };
                states.push(state);
            }
            Err(e) => {
                tracing::error!("Failed to fetch sync state from party {}: {:?}", host, e);
                panic!(
                    "could not get sync state from party {}, trying to restart",
                    host
                );
            }
        }
    }
    Ok(SyncResult::new(my_state.clone(), states))
}

/// Toggle `is_ready_flag` to `true` to signal to other nodes that this node
/// is ready to execute the main server loop.
pub fn set_node_ready(is_ready_flag: Arc<AtomicBool>) {
    tracing::info!("⚓️ ANCHOR: Enable readiness and check all nodes");

    // Set readiness flag to true, i.e. ensure readiness server returns a 200 status code.
    is_ready_flag.store(true, Ordering::SeqCst);
}

/// Awaits until other MPC nodes respond to "ready" queries
/// indicating readiness to execute the main server loop.
pub async fn wait_for_others_ready(config: &Config) -> Result<()> {
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let party_id = config.party_id;
    let ready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(party_id + 2) % 3];
        let mut connected = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;

                if res.is_ok() && res.as_ref().unwrap().status().is_success() {
                    connected[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be ready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        ready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be ready.");
            return Err(eyre!("Timeout waiting for all nodes to be ready."));
        }
    }
    tracing::info!("All nodes are ready.");

    Ok(())
}

/// Assumption This function assumes that each of the nodes have finished indexing the irises before it is called.
pub async fn check_consensus_on_iris_height(config: &Config) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Checking consensus on iris height");
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "height",
    );

    let party_id = config.party_id;
    let height = fetch_height_of_indexed().await;
    let mut heights = [None, None];

    let consensus_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(party_id + 2) % 3];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;

                if let Ok(resp) = res {
                    let height_i = resp.text().await.unwrap().parse::<i64>().unwrap();
                    heights[i] = Some(height_i);
                }
                // If all nodes are connected, notify the main thread.
                if heights.iter().all(|&c| c.is_some()) {
                    return;
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to report height...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        consensus_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to respond to height.");
            bail!("Timeout waiting for all nodes to respond to height.");
        }
    };

    if heights.iter().all(|&height_i| height_i == Some(height)) {
        tracing::info!("All nodes are on height {}", height);
    } else {
        let error_msg = format!(
            "Nodes are on different heights: {:?} respectively. Aborting ..",
            heights
        );
        tracing::info!(error_msg);
        bail!(error_msg)
    }

    Ok(())
}
