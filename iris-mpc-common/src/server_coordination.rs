use crate::config::Config;
use crate::helpers::shutdown_handler::ShutdownHandler;
use crate::helpers::sync::{SyncResult, SyncState};
use crate::helpers::task_monitor::TaskMonitor;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{ensure, Error, OptionExt as _, Result, WrapErr};
use futures::future::try_join_all;
use futures::FutureExt as _;
use itertools::Itertools as _;
use reqwest::Response;
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

pub fn get_check_addresses<S>(hostnames: &[S], ports: &[S], endpoint: &str) -> Vec<String>
where
    S: AsRef<str>,
{
    hostnames
        .iter()
        .zip(ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host.as_ref(), port.as_ref(), endpoint))
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

    // Check other nodes and wait until all nodes are unready.
    let connected_but_unready = try_get_endpoint_all_nodes(config, "ready").await?;

    let all_unready = connected_but_unready
        .iter()
        .all(|resp| resp.status() == StatusCode::SERVICE_UNAVAILABLE);

    ensure!(all_unready, "One or more nodes were not unready.");

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

    let all_health_addresses =
        get_check_addresses(&config.node_hostnames, &config.healthcheck_ports, "health");

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

    tracing::info!("Database store length is: {}", my_state.db_len);

    let connected_and_ready = try_get_endpoint_all_nodes(config, "startup-sync").await?;

    let response_texts_futs: Vec<_> = connected_and_ready
        .into_iter()
        .map(|resp| resp.json())
        .collect();
    let mut all_sync_states: Vec<SyncState> = try_join_all(response_texts_futs).await?;

    all_sync_states.remove(config.party_id);

    Ok(SyncResult::new(my_state.clone(), all_sync_states))
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
    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be ready");

    // Check other nodes and wait until all nodes are ready.
    'outer: loop {
        'retry: {
            let connected_and_ready_res = try_get_endpoint_all_nodes(config, "ready").await;

            if connected_and_ready_res.is_err() {
                break 'retry;
            }

            let connected_and_ready = connected_and_ready_res.unwrap();

            tracing::debug!("{:#?}", &connected_and_ready);
            // connected_and_ready.remove(config.party_id);

            let all_ready = connected_and_ready
                .iter()
                .all(|resp| resp.status().is_success());

            if all_ready {
                break 'outer;
            }
        }
        tracing::debug!("One or more nodes were not ready.  Retrying ..");
    }

    tracing::info!("All nodes are ready.");

    Ok(())
}

/// Assumption This function assumes that each of the nodes have finished indexing the irises before it is called.
pub async fn check_consensus_on_iris_height(config: &Config) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Checking consensus on iris height");
    // Check other nodes and wait until all nodes are ready.

    let iris_heights =
        try_get_endpoint_all_nodes(config, "height-of-graph-genesis-indexation").await?;

    // REVIEW: Merge after height endpoint
    // let height = fetch_height_of_indexed().await;
    let height = 1;
    let response_texts_futs: Vec<_> = iris_heights.into_iter().map(|resp| resp.text()).collect();
    let response_texts = try_join_all(response_texts_futs).await?;
    let response_parses: Result<Vec<_>> = response_texts
        .into_iter()
        .map(|b| b.parse::<i64>().wrap_err("Parse error"))
        .collect();
    let response_i64s = response_parses?;
    let nodes_in_sync = response_i64s.iter().all(|&height_i| height_i == height);

    ensure!(
        nodes_in_sync,
        "One or more nodes were not on the same iris height: {:?}",
        response_i64s
    );

    tracing::info!("All nodes are on height {}", height);

    Ok(())
}

const TIME_BETWEEN_RETRIES: std::time::Duration = Duration::from_secs(1);

pub async fn try_get_endpoint_all_nodes(config: &Config, endpoint: &str) -> Result<Vec<Response>> {
    const NODE_COUNT: usize = 3;
    let full_urls =
        get_check_addresses(&config.node_hostnames, &config.healthcheck_ports, endpoint);
    let node_urls = (0..NODE_COUNT)
        .map(|j| (config.party_id + j) % NODE_COUNT)
        .map(|i| (i, full_urls[i].to_owned()))
        .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
        .map(|(_i, full_url)| full_url);

    let mut handles = Vec::with_capacity(NODE_COUNT);
    let mut rxs = Vec::with_capacity(NODE_COUNT);

    for node_url in node_urls {
        let (tx, rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            loop {
                if let Ok(resp) = reqwest::get(&node_url).await {
                    let _ = tx.send(resp);
                    return;
                }
                tokio::time::sleep(TIME_BETWEEN_RETRIES).await;
            }
        });
        handles.push(handle);
        rxs.push(rx);
    }

    // Wait until timeout
    let all_handles = try_join_all(handles);
    let _all_handles_with_timeout = tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        all_handles,
    )
    .await;

    let msg = "Error occured reading response channels";
    try_join_all(rxs)
        .now_or_never()
        .ok_or_eyre(msg)?
        .inspect_err(|err| {
            tracing::error!("{}: {}", msg, err);
        })
        .wrap_err(msg)
}
