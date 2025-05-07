use axum::{routing::get, Router};
use eyre::Context;
use std::io::{Error as IoError, ErrorKind};

pub fn install_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

pub async fn spawn_healthcheck_server(healthcheck_port: usize) -> Result<()> {
    let app = Router::new().route("/health", get(|| async {})); // Implicit 200 response
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", healthcheck_port))
        .await
        .wrap_err("Healthcheck listener bind error")?;
    axum::serve(listener, app)
        .await
        .wrap_err("healthcheck listener server launch error")?;
    Ok(())
}

pub fn extract_domain(address: &str, remove_protocol: bool) -> Result<String, IoError> {
    // Try to split the address into domain and port parts.
    let mut address = address.trim().to_string();
    if remove_protocol {
        address = address
            .strip_prefix("http://")
            .or_else(|| address.strip_prefix("https://"))
            .unwrap_or(&address)
            .to_string();
    }

    if let Some((domain, _port)) = address.rsplit_once(':') {
        Ok(domain.to_string())
    } else {
        Err(IoError::new(
            ErrorKind::InvalidInput,
            "Invalid address format",
        ))
    }
}
