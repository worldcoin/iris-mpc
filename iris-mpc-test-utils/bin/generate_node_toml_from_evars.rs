use clap::Parser;
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_test_utils::resources::generators::generate_node_config_from_env_vars;
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
    let cfg: NodeConfig = generate_node_config_from_env_vars();
    let cfg_toml = toml::to_string_pretty(&cfg)?;

    // Write to fsys.
    let mut fhandle = fs::File::create(args.path_to_output_file)?;
    fhandle.write_all(cfg_toml.as_bytes())?;

    Ok(())
}
