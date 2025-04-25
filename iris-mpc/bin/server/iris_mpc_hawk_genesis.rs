#![allow(clippy::needless_range_loop)]

use clap::Parser;
use iris_mpc::server::server_main;
use iris_mpc::services::init::initialize_tracing;
use iris_mpc_common::config::{Config, Opt};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // Load .env file(s).
    dotenvy::dotenv().ok();

    // Set config.
    println!("Initialising config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    // Set tracing.
    println!("Initialising tracing");
    let _tracing_shutdown_handle = match initialize_tracing(&config) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    // Invoke main.
    match server_main(config).await {
        Ok(_) => {
            tracing::info!("Server exited normally");
        }
        Err(e) => {
            tracing::error!("Server exited with error: {:?}", e);
            return Err(e);
        }
    }
    Ok(())
}
