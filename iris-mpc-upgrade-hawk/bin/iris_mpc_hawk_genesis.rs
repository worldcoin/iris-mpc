use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::{config::Config, tracing::initialize_tracing, IrisSerialId};
use iris_mpc_cpu::genesis::{log_error, log_info};
use iris_mpc_upgrade_hawk::genesis::exec_main;

/// Process command line arguments.
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

    // Whether to perform a snapshot.
    #[clap(long("perform-snapshot"), default_value = "true")]
    perform_snapshot: bool,
}

/// Process main entry point: performs initial indexation of HNSW graph and optionally
/// creates a db snapshot within AWS RDS cluster.
#[tokio::main]
async fn main() -> Result<()> {
    // Set config.
    println!("Initialising config");
    dotenvy::dotenv().ok();
    let config: Config = Config::load_config("SMPC")?;

    // Set args.
    println!("Initialising args");
    let (height_max, batch_size, perform_snapshot) = parse_args(&config)?;

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
    match exec_main(config, height_max, batch_size, perform_snapshot).await {
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

/// Parses command line arguments: necessary as some typed args need to be
/// passed as optional strings due to CI/CD environment constraints.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
fn parse_args(config: &Config) -> Result<(IrisSerialId, usize, bool)> {
    let args = Args::parse();

    // Parse arg: max indexation height.
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

    // Parse arg: batch size.
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
        config.max_batch_size
    };

    Ok((height_max, batch_size, args.perform_snapshot))
}
