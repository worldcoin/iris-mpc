use clap::Parser;
use rand::{rng, seq::index::sample};
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use std::{fs::File, io::Write, path::Path};

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    // Path to output file.
    #[clap(long("output"))]
    path_to_output_file: String,

    // Maximum range of population to sample.
    #[clap(long("range_max"), default_value = "2000000")]
    range_max: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Struct for JSON encoding.
    #[derive(Serialize, Deserialize)]
    struct Output {
        deleted_serial_ids: Vec<u64>,
    }

    // Set args.
    let args = Args::parse();
    if args.range_max > 20_000_000 {
        return Err("Maximum number should not exceed 2,000,000".into());
    }

    // Generate random selection of integers.
    let mut rng = rng();
    let count = (args.range_max as usize) / 100;
    let indices = sample(&mut rng, args.range_max as usize, count);

    // Convert to 1-based indices and collect into a vector.
    let mut numbers: Vec<u64> = indices.into_iter().map(|i| (i + 1) as u64).collect();
    numbers.sort();

    // Write to file.
    let path = Path::new(&args.path_to_output_file);
    let mut file = File::create(path)?;
    let json_output = json!(Output {
        deleted_serial_ids: numbers
    });
    file.write_all(json_output.to_string().as_bytes())?;

    println!(
        "Successfully wrote {} random numbers to {}",
        count, args.path_to_output_file
    );

    Ok(())
}
