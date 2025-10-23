use axum::{routing::get, Router};
use eyre::{Context, Result};

/// Launch a minimal healthcheck HTTP server that reports readiness.
pub async fn spawn_healthcheck_server(healthcheck_port: usize) -> Result<()> {
    let app = Router::new().route("/health", get(|| async {}));
    let port = u16::try_from(healthcheck_port)
        .wrap_err("Healthcheck port must fit within an unsigned 16-bit integer")?;
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .wrap_err("Healthcheck listener bind error")?;
    axum::serve(listener, app)
        .await
        .wrap_err("Healthcheck listener server launch error")?;
    Ok(())
}
