use clap::Parser;
use eyre::Result;
use iris_mpc_common::{config::Config, tracing::initialize_tracing, IrisSerialId};
use iris_mpc_cpu::genesis::logger;
use iris_mpc_upgrade_hawk::genesis::exec_main;

#[derive(Parser)]
#[allow(non_snake_case)]
#[derive(Debug)]
struct Args {
    // Maximum height of indexation.
    #[clap(long("max-height"), default_value = "5000")]
    max_indexation_height: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set args.
    let args = Args::parse();
    let max_indexation_height: IrisSerialId = args.max_indexation_height.parse().map_err(|_| {
        eprintln!("Error: --max-height argument must be a valid u32.");
        eyre::eyre!("--max-height argument must be a valid u32.")
    })?;

    // Set config.
    println!("Initialising config");
    dotenvy::dotenv().ok();
    let config: Config = Config::load_config("SMPC").unwrap();

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
