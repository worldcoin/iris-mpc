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

/// Parses a batch size configuration string.
///
/// # Formats
/// - `"static:<size>"` - e.g., `"static:100"`
/// - `"dynamic:cap=<cap>,error_rate=<rate>"` - e.g., `"dynamic:cap=500,error_rate=128"`
fn parse_batch_size_config(s: &str) -> Result<BatchSizeConfig> {
    if let Some(size_str) = s.strip_prefix("static:") {
        let size: usize = size_str.parse().map_err(|_| {
            eyre::eyre!(
                "Invalid static batch size '{}'. Expected format: static:<size> (e.g., static:100)",
                size_str
            )
        })?;
        if size == 0 {
            bail!("Static batch size must be greater than 0");
        }
        return Ok(BatchSizeConfig::Static { size });
    }

    if let Some(params_str) = s.strip_prefix("dynamic:") {
        let mut cap: Option<usize> = None;
        let mut error_rate: Option<usize> = None;

        for part in params_str.split(',') {
            let part = part.trim();
            if let Some(val) = part.strip_prefix("cap=") {
                cap = Some(val.parse().map_err(|_| {
                    eyre::eyre!("Invalid cap value '{}'. Expected a positive integer.", val)
                })?);
            } else if let Some(val) = part.strip_prefix("error_rate=") {
                error_rate = Some(val.parse().map_err(|_| {
                    eyre::eyre!(
                        "Invalid error_rate value '{}'. Expected a positive integer.",
                        val
                    )
                })?);
            } else {
                bail!(
                    "Unknown parameter '{}'. Expected 'cap=<value>' or 'error_rate=<value>'.",
                    part
                );
            }
        }

        let cap = cap.ok_or_else(|| {
            eyre::eyre!(
                "Missing 'cap' parameter. Expected format: dynamic:cap=<cap>,error_rate=<rate>"
            )
        })?;
        let error_rate = error_rate.ok_or_else(|| {
            eyre::eyre!(
                "Missing 'error_rate' parameter. Expected format: dynamic:cap=<cap>,error_rate=<rate>"
            )
        })?;

        if cap == 0 {
            bail!("Dynamic batch cap must be greater than 0");
        }
        if error_rate == 0 {
            bail!("Dynamic error_rate must be greater than 0");
        }

        return Ok(BatchSizeConfig::Dynamic { cap, error_rate });
    }

    bail!(
        "Invalid batch size format '{}'. Expected:\n  \
         - static:<size> (e.g., static:100)\n  \
         - dynamic:cap=<cap>,error_rate=<rate> (e.g., dynamic:cap=500,error_rate=128)",
        s
    )
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
    let batch_size_config = parse_batch_size_config(batch_size_arg).map_err(|e| {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_static() {
        let config = parse_batch_size_config("static:100").unwrap();
        assert_eq!(config, BatchSizeConfig::Static { size: 100 });
    }

    #[test]
    fn test_parse_static_zero_fails() {
        let result = parse_batch_size_config("static:0");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_dynamic() {
        let config = parse_batch_size_config("dynamic:cap=500,error_rate=128").unwrap();
        assert_eq!(
            config,
            BatchSizeConfig::Dynamic {
                cap: 500,
                error_rate: 128
            }
        );
    }

    #[test]
    fn test_parse_dynamic_reversed_order() {
        let config = parse_batch_size_config("dynamic:error_rate=128,cap=500").unwrap();
        assert_eq!(
            config,
            BatchSizeConfig::Dynamic {
                cap: 500,
                error_rate: 128
            }
        );
    }

    #[test]
    fn test_parse_dynamic_missing_cap() {
        let result = parse_batch_size_config("dynamic:error_rate=128");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cap"));
    }

    #[test]
    fn test_parse_dynamic_missing_error_rate() {
        let result = parse_batch_size_config("dynamic:cap=500");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("error_rate"));
    }

    #[test]
    fn test_parse_invalid_format() {
        let result = parse_batch_size_config("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_display_static() {
        let config = BatchSizeConfig::Static { size: 100 };
        assert_eq!(config.to_string(), "static:100");
    }

    #[test]
    fn test_display_dynamic() {
        let config = BatchSizeConfig::Dynamic {
            cap: 500,
            error_rate: 128,
        };
        assert_eq!(config.to_string(), "dynamic:cap=500,error_rate=128");
    }

    #[test]
    fn test_roundtrip() {
        for input in ["static:42", "dynamic:cap=1000,error_rate=64"] {
            let config = parse_batch_size_config(input).unwrap();
            let output = config.to_string();
            let reparsed = parse_batch_size_config(&output).unwrap();
            assert_eq!(config, reparsed);
        }
    }
}
