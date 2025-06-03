use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::{config::Config, tracing::initialize_tracing, IrisSerialId};
use iris_mpc_cpu::genesis::{log_error, log_info};
use iris_mpc_upgrade_hawk::genesis::exec_main;

// Default values for batch processing
const DEFAULT_BATCH_SIZE: usize = 0; // Dynamic batch size
const DEFAULT_BATCH_ERROR_RATE: usize = 128; // Default error rate

#[derive(Parser)]
#[allow(non_snake_case)]
#[derive(Debug)]
struct Args {
    // Maximum height of indexation.
    #[clap(long("max-height"))]
    max_indexation_height: Option<String>,

    // Batch size for processing.
    #[clap(long("batch-size"))]
    batch_size: Option<String>,

    // Batch size for processing.
    #[clap(long("batch-size-r"))]
    batch_size_error_rate: Option<String>,

    // Whether to perform a snapshot.
    #[clap(long("perform-snapshot"), default_value = "true")]
    perform_snapshot: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set config.
    println!("Initialising config");
    dotenvy::dotenv().ok();
    let config: Config = Config::load_config("SMPC").unwrap();
    // Set args.
    let args = Args::parse();

    if args.max_indexation_height.is_none() {
        eprintln!("Error: --max-height argument is required.");
        bail!("--max-height argument is required.");
    }
    let max_indexation_height_arg = args.max_indexation_height.as_ref().unwrap();
    let height_max: IrisSerialId = max_indexation_height_arg.parse().map_err(|_| {
        // print the value that is sent to the function
        eprintln!(
            "Error: --max-height argument must be a valid u32. Value: {}",
            max_indexation_height_arg
        );
        eyre::eyre!(
            "--max-height argument must be a valid u32. Value: {}",
            max_indexation_height_arg
        )
    })?;

    let batch_size = if args.batch_size.is_some() {
        let batch_size_arg = args.batch_size.as_ref().unwrap();
        batch_size_arg.parse().map_err(|_| {
            eprintln!(
                "Error: --batch-size argument must be a valid usize. Value: {}",
                batch_size_arg
            );
            eyre::eyre!(
                "--batch-size argument must be a valid usize. Value: {}",
                batch_size_arg
            )
        })?
    } else {
        eprintln!(
            "--batch-size argument not provided, defaulting to {} (dynamic batch size).",
            DEFAULT_BATCH_SIZE
        );
        DEFAULT_BATCH_SIZE
    };

    let batch_size_error_rate = if args.batch_size_error_rate.is_some() {
        let batch_size_error_rate_arg = args.batch_size_error_rate.as_ref().unwrap();
        batch_size_error_rate_arg.parse().map_err(|_| {
            eprintln!(
                "Error: --batch-size-r argument must be a valid usize. Value: {}",
                batch_size_error_rate_arg
            );
            eyre::eyre!(
                "--batch-size-r argument must be a valid usize. Value: {}",
                batch_size_error_rate_arg
            )
        })?
    } else {
        eprintln!(
            "--batch-size-r argument not provided, defaulting to {} (error rate).",
            DEFAULT_BATCH_ERROR_RATE
        );
        DEFAULT_BATCH_ERROR_RATE
    };

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
    match exec_main(
        config,
        height_max,
        batch_size,
        batch_size_error_rate,
        args.perform_snapshot,
    )
    .await
    {
        Ok(_) => {
            log_info("Server", "Exited normally".to_string());
        }
        Err(err) => {
            log_error("Server", format!("Server exited with error: {:?}", err));
            return Err(err);
        }
    }
    Ok(())
}
