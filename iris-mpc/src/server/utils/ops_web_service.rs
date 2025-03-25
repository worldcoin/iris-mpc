use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
use eyre::{Result, WrapErr};
use iris_mpc_common::{
    config::Config,
    helpers::{shutdown_handler::ShutdownHandler, sync::SyncState},
};
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct ReadyProbeResponse {
    pub image_name: String,
    pub uuid: String,
    pub shutting_down: bool,
}

pub(crate) async fn get_spinup_web_service_future(
    config: Config,
    sync_state: SyncState,
    shutdown_handler: Arc<ShutdownHandler>,
    is_ready_flag: Arc<AtomicBool>,
) -> impl Future<Output = Result<()>> + Send {
    let uuid = uuid::Uuid::new_v4().to_string();

    // Set fixed respones.
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

    // Spinup server.
    let sync_state = sync_state.clone();
    async move {
        // Generate a random UUID for each run.
        let app = Router::new()
            .route(
                "/health",
                get(move || {
                    let shutdown_handler_clone = Arc::clone(&shutdown_handler);
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
                get(move || async move { serde_json::to_string(&sync_state).unwrap() }),
            );

        let listener = tokio::net::TcpListener::bind(format!(
            "0.0.0.0:{}",
            config.hawk_server_healthcheck_port
        ))
        .await
        .wrap_err("Failed to bind to healthcheck listener");

        axum::serve(listener.unwrap(), app)
            .await
            .wrap_err("Failed to launch healthcheck server")
    }
}
