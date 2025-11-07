use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{routing::get, Json, Router};
use eyre::{Context, Result};
use iris_mpc_common::server_coordination::ReadyProbeResponse;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone)]
pub struct HealthServerState {
    pub is_ready: Arc<AtomicBool>,
    pub is_shutting_down: Arc<AtomicBool>,
    pub image_name: Arc<String>,
    pub uuid: Arc<String>,
}

impl HealthServerState {
    pub fn new(
        is_ready: Arc<AtomicBool>,
        is_shutting_down: Arc<AtomicBool>,
        image_name: Arc<String>,
        uuid: Arc<String>,
    ) -> Self {
        Self {
            is_ready,
            is_shutting_down,
            image_name,
            uuid,
        }
    }
}

pub async fn spawn_healthcheck_server_with_state(
    healthcheck_port: String,
    state: HealthServerState,
) -> Result<()> {
    let health_state = state.clone();
    let ready_state = state;

    let app = Router::new()
        .route(
            "/health",
            get(move || {
                let state = health_state.clone();
                async move {
                    let response = ReadyProbeResponse {
                        image_name: state.image_name.as_ref().clone(),
                        uuid: state.uuid.as_ref().clone(),
                        shutting_down: state.is_shutting_down.load(Ordering::SeqCst),
                    };
                    Json(response)
                }
            }),
        )
        .route(
            "/ready",
            get(move || {
                let state = ready_state.clone();
                async move {
                    if state.is_ready.load(Ordering::SeqCst) {
                        StatusCode::OK.into_response()
                    } else {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    }
                }
            }),
        );

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", healthcheck_port))
        .await
        .wrap_err("Healthcheck listener bind error")?;
    axum::serve(listener, app)
        .await
        .wrap_err("Healthcheck listener server launch error")?;
    Ok(())
}
