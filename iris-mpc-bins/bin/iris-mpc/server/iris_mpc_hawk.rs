#![allow(clippy::needless_range_loop)]

use clap::Parser;
use eyre::Result;
use iris_mpc::server::server_main;
use iris_mpc_common::config::{Config, Opt};
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_common::{get_node_zero_cores, restrict_to_node_zero};
use std::process::exit;

fn main() -> Result<()> {
    restrict_to_node_zero();
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    // Build the Tokio runtime first so any telemetry exporters that spawn tasks have a runtime.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(get_node_zero_cores())
        .on_thread_start(move || {
            restrict_to_node_zero();
        })
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("Init tracing");
        let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
            Ok(handle) => handle,
            Err(e) => {
                eprintln!("Failed to initialize tracing: {:?}", e);
                return Err(e);
            }
        };

        match server_main(config).await {
            Ok(_) => {
                tracing::info!("Server exited normally");
            }
            Err(e) => {
                tracing::error!("Server exited with error: {:?}", e);
                exit(1);
            }
        }
        Ok(())
    })
}
