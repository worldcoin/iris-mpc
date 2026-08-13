#![allow(clippy::needless_range_loop)]
#![recursion_limit = "256"]

use clap::Parser;
use eyre::Result;
use iris_mpc::server::linear_scan_server_main;
use iris_mpc_common::config::{Config, Opt};
use iris_mpc_common::helpers::numactl;
use iris_mpc_common::tracing::initialize_tracing;
use std::process::exit;

fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("Init config");
    let mut config: Config = Config::load_config("SMPC").unwrap();
    config.overwrite_defaults_with_cli_args(Opt::parse());

    numactl::init(config.separate_tokio_cores_per_node);
    numactl::restrict_tokio_runtime();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(numactl::get_tokio_worker_threads())
        .on_thread_start(numactl::restrict_tokio_runtime)
        .enable_all()
        .build()
        .unwrap();

    let party_id = config.party_id;
    runtime.block_on(async {
        println!("Init tracing");
        let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
            Ok(handle) => handle,
            Err(e) => {
                eprintln!("Failed to initialize tracing: {e:?}");
                return Err(e);
            }
        };

        if let Err(e) = linear_scan_server_main(config).await {
            tracing::error!("EXIT_SUMMARY party={} error=\"{:?}\"", party_id, e);
            exit(1);
        }
        tracing::info!("Server exited normally");
        Ok(())
    })
}
