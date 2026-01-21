use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::{config::Config, tracing::initialize_tracing, IrisSerialId};
use iris_mpc_cpu::genesis::{log_error, log_info, BatchSizeConfig};
use iris_mpc_upgrade_hawk::genesis::{exec, ExecutionArgs};

#[derive(Parser)]
struct Args {
    // Maximum height of indexation.
    #[clap(long("max-height"))]
    max_indexation_height: Option<String>,

    /// Batch size configuration (required).
    ///
    /// Format:
    ///   - static:<size>                           (e.g., "static:100")
    ///   - dynamic:cap=<cap>,error_rate=<rate>     (e.g., "dynamic:cap=500,error_rate=128")
    #[clap(long("batch-size"))]
    batch_size: Option<String>,

    // Whether to perform a snapshot.
    #[clap(long("perform-snapshot"))]
    perform_snapshot: Option<String>,

    // User backup as source.
    #[clap(long("use-backup-as-source"))]
    use_backup_as_source: Option<String>,
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
    let args = parse_args()?;

    // Set tracing.
    println!("Initialising tracing");
    let _tracing_shutdown_handle = match initialize_tracing(config.service.clone()) {
        Ok(handle) => handle,
        Err(e) => {
            eprintln!("Failed to initialize tracing: {:?}", e);
            return Err(e);
        }
    };

    // Invoke main.
    match exec(args, config).await {
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

/// Parses command line arguments.
///
/// Necessary as within CI/CD environments typed args need to be passed as optional strings.
fn parse_args() -> Result<ExecutionArgs> {
    let args = Args::parse();

    // Arg: max indexation height.
    if args.max_indexation_height.is_none() {
        eprintln!("Error: --max-height argument is required.");
        bail!("--max-height argument is required.");
    }
    let max_indexation_height_arg = args.max_indexation_height.as_ref().unwrap();
    let max_indexation_id: IrisSerialId = max_indexation_height_arg.parse().map_err(|_| {
        eprintln!(
            "Error: --max-height argument must be a valid u32. Value: {}",
            max_indexation_height_arg
        );
        eyre::eyre!(
            "--max-height argument must be a valid u32. Value: {}",
            max_indexation_height_arg
        )
    })?;

    // Arg: batch size configuration.
    if args.batch_size.is_none() {
        eprintln!("Error: --batch-size argument is required.");
        eprintln!();
        eprintln!("Expected format:");
        eprintln!("  --batch-size 'static:<size>'");
        eprintln!("  --batch-size 'dynamic:cap=<cap>,error_rate=<rate>'");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  --batch-size 'static:100'");
        eprintln!("  --batch-size 'dynamic:cap=500,error_rate=128'");
        bail!("--batch-size argument is required.");
    }
    let batch_size_arg = args.batch_size.as_ref().unwrap();
    let batch_size_config = BatchSizeConfig::parse(batch_size_arg).map_err(|e| {
        eprintln!("Error parsing --batch-size: {}", e);
        eprintln!();
        eprintln!("Expected format:");
        eprintln!("  --batch-size 'static:<size>'");
        eprintln!("  --batch-size 'dynamic:cap=<cap>,error_rate=<rate>'");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  --batch-size 'static:100'");
        eprintln!("  --batch-size 'dynamic:cap=500,error_rate=128'");
        e
    })?;

    // Arg: perform snapshot.
    let perform_snapshot = if args.perform_snapshot.is_some() {
        let perform_snapshot_arg = args.perform_snapshot.as_ref().unwrap();
        perform_snapshot_arg.parse().map_err(|_| {
            eprintln!(
                "Error: --perform-snapshot argument must be a valid boolean. Value: {}",
                perform_snapshot_arg
            );
            eyre::eyre!(
                "--perform-snapshot argument must be a valid boolean. Value: {}",
                perform_snapshot_arg
            )
        })?
    } else {
        true
    };

    // Arg: use_backup_as_source (parse as string, convert to bool for ExecutionArgs).
    let use_backup_as_source = if args.use_backup_as_source.is_some() {
        let use_backup_as_source_args = args.use_backup_as_source.as_ref().unwrap();
        use_backup_as_source_args.parse().map_err(|_| {
            eprintln!(
                "Error: --use-backup-as-source argument must be a valid boolean. Value: {}",
                use_backup_as_source_args
            );
            eyre::eyre!(
                "--use-backup-as-source argument must be a valid boolean. Value: {}",
                use_backup_as_source_args
            )
        })?
    } else {
        eprintln!("--use-backup-as-source argument not provided, defaulting to false.");
        false
    };

    Ok(ExecutionArgs::new(
        batch_size_config,
        max_indexation_id,
        perform_snapshot,
        use_backup_as_source,
    ))
}
