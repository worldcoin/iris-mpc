use crate::config::Config;
use crate::helpers::batch_sync::get_own_batch_sync_entries;
use ampc_server_utils::{shutdown_handler::ShutdownHandler, ReadyProbeResponse, TaskMonitor};
use axum::extract::Query;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{Error, WrapErr};
use pprof::protos::Message;
use pprof::ProfilerGuardBuilder;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::sync::Mutex;
use tokio::time::sleep as tokio_sleep;

#[derive(Debug, Deserialize)]
struct BatchSyncQuery {
    batch_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct BatchSyncSharedState {
    pub batch_id: u64,
    pub messages_to_poll: u32,
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
    let server_coord_config = &config
        .server_coordination
        .clone()
        .unwrap_or_else(|| panic!("Server coordination config is required for server operation"));

    let _health_check_abort = task_monitor.spawn({
        let uuid = uuid::Uuid::new_v4().to_string();
        let is_ready_flag = Arc::clone(&is_ready_flag);
        let ready_probe_response = ReadyProbeResponse {
            image_name: server_coord_config.image_name.clone(),
            shutting_down: false,
            uuid: uuid.clone(),
        };
        let ready_probe_response_shutdown = ReadyProbeResponse {
            image_name: server_coord_config.image_name.clone(),
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
