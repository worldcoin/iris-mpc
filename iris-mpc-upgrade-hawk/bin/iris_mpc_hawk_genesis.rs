use clap::Parser;
use eyre::Result;
use iris_mpc_common::{
    config::{Config, Opt},
    tracing::initialize_tracing,
    IrisSerialId,
};
use iris_mpc_cpu::genesis::logger;
use iris_mpc_upgrade_hawk::genesis::exec_main;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    // Maximum height of indexation.
    #[clap(long("max-height"))]
    max_indexation_height: Option<IrisSerialId>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set args.
    let args = Args::parse();
    let max_indexation_height = args.max_indexation_height;

    // How can i throw an error here if the max_indexation_height is not set?
    if max_indexation_height.is_none() {
        eprintln!("Error: --max-height argument is required.");
        bail!("--max-height argument is required.");
    }
    let max_indexation_height = max_indexation_height.unwrap();

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
            logger::log_info("Server", "Exited normally".to_string());
        }
        Err(err) => {
            logger::log_error("Server", format!("Server exited with error: {:?}", err));
            return Err(err);
        }
    }
    Ok(())
}
