use clap::Parser;
use iris_mpc_common::config::Config as NodeConfig;
use std::fs;
use std::io::Write;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    // Path to output file.
    #[clap(long("output"))]
    path_to_output_file: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set args.
    let args = Args::parse();

    // Set config.
    dotenvy::dotenv().ok();
    let cfg: NodeConfig = NodeConfig::load_config("SMPC")?;
    let cfg = toml::to_string_pretty(&cfg)?;

    // Write to fsys.
    let mut fhandle = fs::File::create(args.path_to_output_file)?;
    fhandle.write_all(cfg.as_bytes())?;

    Ok(())
}
