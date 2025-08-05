use clap::Parser;
use iris_mpc_cpu::genesis::utils::aws::IrisDeletionsForS3;
use iris_mpc_test_utils::resources::generators::generate_iris_deletions;
use serde_json::{self, json};
use std::{fs::File, io::Write, path::Path};

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    // Path to output file.
    #[clap(long("output"))]
    path_to_output_file: String,

    // Maximum range of population to sample.
    #[clap(long("count"), default_value = "1000")]
    n_deletions: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set args.
    let args = Args::parse();
    if args.n_deletions > 300_000 {
        return Err("Maximum number should not exceed 300,000".into());
    }

    // Generate Iris serial identifiers.
    let mut indices = generate_iris_deletions(args.n_deletions);
    indices.sort();

    // Write to file.
    let path = Path::new(&args.path_to_output_file);
    let mut file = File::create(path).unwrap();
    let json_output = json!(IrisDeletionsForS3 {
        deleted_serial_ids: indices.clone()
    });
    file.write_all(json_output.to_string().as_bytes())?;

    println!(
        "Successfully wrote {} random numbers to {}",
        indices.len(),
        args.path_to_output_file
    );

    Ok(())
}
