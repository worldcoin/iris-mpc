use clap::Parser;
use eyre::Result;
use iris_mpc_common::config::{Config, Opt};
use iris_mpc_common::tracing::initialize_tracing;
use iris_mpc_upgrade_hawk::genesis::exec_main;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    // Maximum height of indexation.
    #[clap(long("max-height"))]
    max_indexation_height: Option<u64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set args.
    let args = Args::parse();
    let max_indexation_height = args.max_indexation_height.unwrap_or(0);

    // Set config.
    println!("Initialising config");
    dotenvy::dotenv().ok();
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
    match exec_main(config, max_indexation_height).await {
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
