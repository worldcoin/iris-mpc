use crate::config::Config;
use crate::helpers::batch_sync::get_own_batch_sync_entries;
use crate::helpers::shutdown_handler::ShutdownHandler;
use crate::helpers::task_monitor::TaskMonitor;
use axum::extract::Query;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{bail, ensure, Error, OptionExt as _, Result, WrapErr};
use futures::future::try_join_all;
use futures::FutureExt as _;
use itertools::Itertools as _;
use pprof::protos::Message;
use pprof::ProfilerGuardBuilder;
use reqwest::Response;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Duration as StdDuration, Instant};
use tokio::sync::{oneshot, Mutex};
use tokio::time::sleep as tokio_sleep;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReadyProbeResponse {
    pub image_name: String,
    pub uuid: String,
    pub shutting_down: bool,
}

#[derive(Debug, Deserialize)]
struct BatchSyncQuery {
    batch_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct BatchSyncSharedState {
    pub batch_id: u64,
    pub messages_to_poll: u32,
}

#[derive(Debug, Clone)]
pub struct CoordinationSettings<'a> {
    pub party_id: usize,
    pub node_hostnames: &'a [String],
    pub healthcheck_ports: &'a [String],
    pub image_name: &'a str,
    pub http_query_retry_delay_ms: u64,
    pub startup_sync_timeout_secs: u64,
    pub heartbeat_interval_secs: u64,
    pub heartbeat_initial_retries: u64,
}

impl<'a> CoordinationSettings<'a> {
    pub fn from_config(config: &'a Config) -> Self {
        Self {
            party_id: config.party_id,
            node_hostnames: &config.node_hostnames,
            healthcheck_ports: &config.healthcheck_ports,
            image_name: &config.image_name,
            http_query_retry_delay_ms: config.http_query_retry_delay_ms,
            startup_sync_timeout_secs: config.startup_sync_timeout_secs,
            heartbeat_interval_secs: config.heartbeat_interval_secs,
            heartbeat_initial_retries: config.heartbeat_initial_retries,
        }
    }
}

// Returns a new task monitor.
pub fn init_task_monitor() -> TaskMonitor {
    tracing::info!("Preparing task monitor");
    TaskMonitor::new()
}

// ---- Optional pprof HTTP routes ----
#[derive(Debug, Deserialize)]
struct PprofQuery {
    seconds: Option<u64>,
    frequency: Option<i32>,
}

fn pprof_routes() -> Router {
    Router::new()
        .route(
            "/pprof/flame",
            get(|Query(q): Query<PprofQuery>| async move {
                let seconds = q.seconds.unwrap_or(30).min(300);
                let frequency = q.frequency.unwrap_or(99).clamp(1, 1000);
                let guard = ProfilerGuardBuilder::default()
                    .frequency(frequency)
                    .build()
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                tokio_sleep(StdDuration::from_secs(seconds)).await;
                match guard.report().build() {
                    Ok(report) => {
                        let mut svg = Vec::new();
                        if let Err(e) = report.flamegraph(&mut svg) {
                            return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                        }
                        Ok((
                            StatusCode::OK,
                            [(header::CONTENT_TYPE, "image/svg+xml")],
                            svg,
                        ))
                    }
                    Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
                }
            }),
        )
        .route(
            "/pprof/profile",
            get(|Query(q): Query<PprofQuery>| async move {
                let seconds = q.seconds.unwrap_or(30).min(300);
                let frequency = q.frequency.unwrap_or(99).clamp(1, 1000);
                let guard = ProfilerGuardBuilder::default()
                    .frequency(frequency)
                    .build()
                    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
                tokio_sleep(StdDuration::from_secs(seconds)).await;
                match guard.report().build() {
                    Ok(report) => match report.pprof() {
                        Ok(profile) => {
                            let mut buf = Vec::new();
                            if let Err(e) = profile.encode(&mut buf) {
                                return Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string()));
                            }
                            Ok((
                                StatusCode::OK,
                                [(header::CONTENT_TYPE, "application/octet-stream")],
                                buf,
                            ))
                        }
                        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
                    },
                    Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
                }
            }),
        )
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
pub async fn start_coordination_server<T>(
    config: &Config,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    my_state: &T,
    batch_sync_shared_state: Arc<Mutex<BatchSyncSharedState>>,
) -> Arc<AtomicBool>
where
    T: Serialize + DeserializeOwned + Clone + Send + 'static,
{
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
                // Optional: expose pprof endpoints when built with `profiling` feature
                // - /pprof/flame?seconds=30&frequency=99 returns an SVG flamegraph collected on-demand
                // - /pprof/profile?seconds=30&frequency=99 returns a pprof protobuf profile
                .merge(pprof_routes())
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
                    "/batch-sync-state",
                    get(move |Query(params): Query<BatchSyncQuery>| {
                        let batch_sync_shared_state = batch_sync_shared_state.clone();
                        async move {
                            let shared_state = batch_sync_shared_state.lock().await;

                            // Check if the requested batch_id matches our cached batch_id
                            if params.batch_id != shared_state.batch_id {
                                return (
                                    StatusCode::CONFLICT,
                                    format!(
                                        "Batch ID mismatch: requested {}, current {}",
                                        params.batch_id, shared_state.batch_id
                                    ),
                                )
                                    .into_response();
                            }

                            // Return the cached state
                            let batch_sync_state = crate::helpers::batch_sync::BatchSyncState {
                                messages_to_poll: shared_state.messages_to_poll,
                                batch_id: shared_state.batch_id,
                            };

                            match serde_json::to_string(&batch_sync_state) {
                                Ok(body) => (StatusCode::OK, body).into_response(),
                                Err(e) => {
                                    tracing::error!(
                                        "Failed to serialize batch sync state: {:?}",
                                        e
                                    );
                                    (
                                        StatusCode::INTERNAL_SERVER_ERROR,
                                        format!("Serialization error: {}", e),
                                    )
                                        .into_response()
                                }
                            }
                        }
                    }),
                )
                .route(
                    "/batch-sync-entries",
                    get(move || async move {
                        let own_batch_sync_entries = get_own_batch_sync_entries().await;
                        match serde_json::to_string(&own_batch_sync_entries) {
                            Ok(body) => (StatusCode::OK, body).into_response(),
                            Err(e) => {
                                tracing::error!("Failed to serialize batch sync entries: {:?}", e);
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Serialization error: {}", e),
                                )
                                    .into_response()
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
    wait_for_others_unready_with_settings(&CoordinationSettings::from_config(config)).await
}

pub async fn wait_for_others_unready_with_settings(
    settings: &CoordinationSettings<'_>,
) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");

    let connected_but_unready = try_get_endpoint_other_nodes_with_settings(settings, "ready")?;
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
    let settings = CoordinationSettings::from_config(config);
    init_heartbeat_task_with_settings(&settings, task_monitor, shutdown_handler).await
}

pub async fn init_heartbeat_task_with_settings(
    settings: &CoordinationSettings<'_>,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);

    let all_health_addresses = get_check_addresses(
        settings.node_hostnames,
        settings.healthcheck_ports,
        "health",
    );

    let party_id = settings.party_id;
    let image_name = settings.image_name.to_string();
    let heartbeat_initial_retries = settings.heartbeat_initial_retries;
    let heartbeat_interval_secs = settings.heartbeat_interval_secs;

    let heartbeat_shutdown_handler = Arc::clone(shutdown_handler);
    let _heartbeat = task_monitor.spawn(async move {
        let next_node = &all_health_addresses[(party_id + 1) % 3];
        let prev_node = &all_health_addresses[(party_id + 2) % 3];
        let mut last_response = [String::default(), String::default()];
        let mut connected = [false, false];
        let mut retries = [0, 0];
        // Track consecutive failures
        let mut consecutive_failures = [0, 0];
        // Number of consecutive failures before triggering shutdown
        const MAX_CONSECUTIVE_FAILURES: u32 = 3;

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;
                if res.is_err() || !res.as_ref().unwrap().status().is_success() {
                    tracing::warn!(
                        "Node {} did not respond with success, response: {:?}",
                        host,
                        res
                    );
                    // If it's the first time after startup, we allow a few retries to let the other
                    // nodes start up as well.
                    if last_response[i] == String::default()
                        && retries[i] < heartbeat_initial_retries
                    {
                        retries[i] += 1;
                        tracing::warn!("Node {} did not respond with success, retrying...", host);
                        continue;
                    }

                    // if the nodes are still starting up and they get a failure - we can panic and
                    // not start graceful shutdown
                    // we ignore consecutive failures for the initial response
                    if last_response[i] == String::default() {
                        panic!(
                            "Node {} did not respond with success during heartbeat init phase, killing server...",
                            host
                        );
                    }

                    consecutive_failures[i] += 1;
                    tracing::warn!(
                        "Node {} failed health check {} times consecutively",
                        host,
                        consecutive_failures[i]
                    );

                    // Only trigger shutdown after multiple consecutive failures
                    if consecutive_failures[i] >= MAX_CONSECUTIVE_FAILURES {
                        tracing::error!(
                            "Node {} has failed {} consecutive health checks, starting graceful shutdown",
                            host,
                            MAX_CONSECUTIVE_FAILURES
                        );

                        if !heartbeat_shutdown_handler.is_shutting_down() {
                            heartbeat_shutdown_handler.trigger_manual_shutdown();
                            tracing::error!(
                                "Node {} has failed consecutive health checks, therefore graceful shutdown has been triggered",
                                host
                            );
                        } else {
                            tracing::info!("Node {} has already started graceful shutdown.", host);
                        }
                    }
                    continue;
                }

                // Reset consecutive failures counter on successful health check
                consecutive_failures[i] = 0;

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
                    tracing::debug!("Heartbeat: Node {} is healthy", host);
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
pub async fn get_others_sync_state<State>(config: &Config) -> Result<Vec<State>>
where
    State: DeserializeOwned + Clone,
{
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");

    let connected_and_ready = try_get_endpoint_other_nodes(config, "startup-sync").await?;

    let response_texts_futs: Vec<_> = connected_and_ready
        .into_iter()
        .map(|resp| resp.json())
        .collect();
    let sync_states: Vec<State> = try_join_all(response_texts_futs).await?;

    Ok(sync_states)
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
    wait_for_others_ready_with_settings(&CoordinationSettings::from_config(config)).await
}

pub async fn wait_for_others_ready_with_settings(
    settings: &CoordinationSettings<'_>,
) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be ready");

    // Check other nodes and wait until all nodes are ready.
    'outer: loop {
        'retry: {
            let connected_and_ready_res =
                try_get_endpoint_other_nodes_with_settings(settings, "ready").await;

            if connected_and_ready_res.is_err() {
                break 'retry;
            }

            let connected_and_ready = connected_and_ready_res.unwrap();

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

    validate_peer_health_with_settings(settings).await?;

    Ok(())
}

/// Retrieve outputs from a healthcheck endpoint from all other server nodes.
///
/// Upon failure, retries with wait duration `config.http_query_retry_delay_ms`
/// between attempts, until `config.startup_sync_timeout_secs` seconds have elapsed.
pub async fn try_get_endpoint_other_nodes(
    config: &Config,
    endpoint: &str,
) -> Result<Vec<Response>> {
    try_get_endpoint_other_nodes_with_settings(&CoordinationSettings::from_config(config), endpoint)
        .await
}

pub async fn try_get_endpoint_other_nodes_with_settings(
    settings: &CoordinationSettings<'_>,
    endpoint: &str,
) -> Result<Vec<Response>> {
    const NODE_COUNT: usize = 3;
    let full_urls = get_check_addresses(
        settings.node_hostnames,
        settings.healthcheck_ports,
        endpoint,
    );
    let node_urls = (1..NODE_COUNT)
        .map(|j| (settings.party_id + j) % NODE_COUNT)
        .map(|i| (i, full_urls[i].to_owned()))
        .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
        .map(|(_i, full_url)| full_url);

    let mut handles = Vec::with_capacity(NODE_COUNT - 1);
    let mut rxs = Vec::with_capacity(NODE_COUNT - 1);

    let retry_duration = Duration::from_millis(settings.http_query_retry_delay_ms);
    for node_url in node_urls {
        let (tx, rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            loop {
                if let Ok(resp) = reqwest::get(&node_url).await {
                    let _ = tx.send(resp);
                    return;
                }
                tokio::time::sleep(retry_duration).await;
            }
        });
        handles.push(handle);
        rxs.push(rx);
    }

    // Wait until timeout
    let all_handles = try_join_all(handles);
    let _all_handles_with_timeout = tokio::time::timeout(
        Duration::from_secs(settings.startup_sync_timeout_secs),
        all_handles,
    )
    .await;

    let msg = "Error occurred reading response channels";
    try_join_all(rxs)
        .now_or_never()
        .ok_or_eyre(msg)?
        .inspect_err(|err| {
            tracing::error!("{}: {}", msg, err);
        })
        .wrap_err(msg)
}

fn other_node_endpoints_with_settings(
    settings: &CoordinationSettings<'_>,
    endpoint: &str,
) -> Result<Vec<String>> {
    if settings.node_hostnames.is_empty() || settings.healthcheck_ports.is_empty() {
        return Ok(Vec::new());
    }

    ensure!(
        settings.node_hostnames.len() == settings.healthcheck_ports.len(),
        "node_hostnames and healthcheck_ports must have the same length"
    );
    ensure!(
        settings.party_id < settings.node_hostnames.len(),
        "party_id {} out of bounds for node hostnames length {}",
        settings.party_id,
        settings.node_hostnames.len()
    );

    let endpoints = settings
        .node_hostnames
        .iter()
        .zip(settings.healthcheck_ports.iter())
        .enumerate()
        .filter(|(idx, _)| *idx != settings.party_id)
        .map(|(_, (host, port))| format!("http://{}:{}/{}", host, port, endpoint))
        .collect();

    Ok(endpoints)
}
async fn validate_peer_health_with_settings(settings: &CoordinationSettings<'_>) -> Result<()> {
    let endpoints = other_node_endpoints_with_settings(settings, "health")?;
    if endpoints.is_empty() {
        return Ok(());
    }

    for endpoint in endpoints {
        match reqwest::get(&endpoint).await {
            Ok(response) if response.status().is_success() => {
                match response.json::<ReadyProbeResponse>().await {
                    Ok(probe) => {
                        if !settings.image_name.is_empty()
                            && !probe.image_name.is_empty()
                            && settings.image_name != probe.image_name
                        {
                            tracing::warn!(
                                url = endpoint,
                                remote_image = probe.image_name,
                                local_image = %settings.image_name,
                                "Peer is running a different image"
                            );
                        }
                        if probe.shutting_down {
                            tracing::warn!(url = endpoint, "Peer is reporting shutdown state");
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            url = endpoint,
                            error = ?err,
                            "Failed to parse ReadyProbeResponse"
                        );
                    }
                }
            }
            Ok(response) => {
                tracing::warn!(
                    url = endpoint,
                    status = ?response.status(),
                    "Unexpected status from peer health endpoint"
                );
            }
            Err(err) => {
                tracing::warn!(
                    url = endpoint,
                    error = ?err,
                    "Failed to query peer health endpoint"
                );
            }
        }
    }

    Ok(())
}
