use axum::{http::StatusCode, routing::get, Router};
use clap::Parser;
use eyre::{Context, Result};
use iris_mpc_anon_stats_server::{
    config::AnonStatsServerConfig, config::Opt, spawn_healthcheck_server,
};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::tracing::initialize_tracing;

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

    let app = Router::new().route("/", get(root_handler));
    let listener = tokio::net::TcpListener::bind(config.bind_addr)
        .await
        .wrap_err_with(|| format!("Failed to bind HTTP server on {}", config.bind_addr))?;
    tracing::info!("HTTP server listening on {}", config.bind_addr);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .wrap_err("HTTP server encountered an unrecoverable error")?;

    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!("Failed to install CTRL+C handler: {:?}", err);
    }
}

async fn root_handler() -> StatusCode {
    StatusCode::NO_CONTENT
}
