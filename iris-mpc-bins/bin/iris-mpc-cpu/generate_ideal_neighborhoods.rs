use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

use clap::{Parser, ValueEnum};
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::naive_knn_plaintext::{Engine, EngineChoice, KNNResult},
    py_bindings::plaintext_store::Base64IrisCode,
};
use metrics::IntoF64;

use serde::{Deserialize, Serialize};
use serde_json::Deserializer;
use std::time::Instant;

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
enum IrisSelection {
    All,
    Even,
    Odd,
}

/// A struct to hold the metadata stored in the first line of the results file.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ResultsHeader {
    iris_selection: IrisSelection,
    num_irises: usize,
    k: usize,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of irises to process
    #[arg(long, default_value_t = 1000)]
    num_irises: usize,

    /// Number of threads to use
    #[arg(long, default_value_t = 1)]
    num_threads: usize,

    /// Path to the iris codes file
    #[arg(long, default_value = "iris-mpc-bins/data/store.ndjson")]
    path_to_iris_codes: PathBuf,

    /// The k for k-NN
    #[arg(long)]
    k: usize,

    /// Path to the results file
    #[arg(long)]
    results_file: PathBuf,

    /// Selection of irises to process
    #[arg(long, value_enum, default_value_t = IrisSelection::All)]
    irises_selection: IrisSelection,

    /// Selection of irises to process
    #[arg(long, value_enum, default_value_t = EngineChoice::NaiveFHD)]
    engine_choice: EngineChoice,
}
#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (num_already_processed, nodes) = match File::open(&args.results_file) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            // 1. Read and validate the header line
            let header_line = match lines.next() {
                Some(Ok(line)) => line,
                Some(Err(e)) => {
                    eprintln!("Error: Could not read header from results file: {}", e);
                    std::process::exit(1);
                }
                None => {
                    eprintln!(
                        "Error: Results file '{}' is empty. Please fix or delete it.",
                        args.results_file.display()
                    );
                    std::process::exit(1);
                }
            };

            let file_header: ResultsHeader = match serde_json::from_str(&header_line) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("Error: Could not parse header in results file: {}", e);
                    eprintln!(
                        " -> Please fix or delete the file '{}' and restart.",
                        args.results_file.display()
                    );
                    std::process::exit(1);
                }
            };

            // 2. Check for configuration mismatches with a single comparison
            let expected_header = ResultsHeader {
                iris_selection: args.irises_selection,
                num_irises: args.num_irises,
                k: args.k,
            };

            if file_header != expected_header {
                eprintln!("Error: Mismatch in results file configuration.");
                eprintln!(" -> Expected parameters: {:?}", expected_header);
                eprintln!(" -> Parameters found in file: {:?}", file_header);
                eprintln!(" -> Please use a different results file or adjust the command-line arguments to match.");
                std::process::exit(1);
            }

            // 3. Process the rest of the lines as KNN results
            let results: Result<Vec<KNNResult>, _> = lines
                .map(|line_result| {
                    let line = line_result.map_err(|e| e.to_string())?;
                    serde_json::from_str::<KNNResult>(&line).map_err(|e| e.to_string())
                })
                .collect();

            let deserialized_results = match results {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("Error: Failed to deserialize a result from the file.");
                    eprintln!("It may be corrupted from an abrupt shutdown.");
                    eprintln!(" -> Error details: {}", e);
                    eprintln!(
                        " -> Please fix or delete the file '{}' and restart.",
                        args.results_file.display()
                    );
                    std::process::exit(1);
                }
            };

            let nodes: Vec<usize> = deserialized_results
                .into_iter()
                .map(|result| result.node as usize)
                .collect();
            (nodes.len(), nodes)
        }
        Err(_) => {
            // File doesn't exist, create it and write the header.
            let mut file = File::create(&args.results_file).expect("Unable to create results file");
            let header = ResultsHeader {
                iris_selection: args.irises_selection,
                num_irises: args.num_irises,
                k: args.k,
            };
            let header_str =
                serde_json::to_string(&header).expect("Failed to serialize ResultsHeader");
            writeln!(file, "{}", header_str).expect("Failed to write header to new results file");
            (0, Vec::new())
        }
    };

    if num_already_processed > 0 {
        let expected_nodes: Vec<usize> = (1..num_already_processed + 1).collect();
        if nodes != expected_nodes {
            eprintln!(
                "Error: The result nodes in the file are not a contiguous sequence from 1 to N."
            );
            eprintln!(
                " -> Please fix or delete the file '{}' and restart.",
                args.results_file.display()
            );
            std::process::exit(1);
        }
    }

    let path_to_iris_codes = args.path_to_iris_codes;
    assert!(args.num_irises > args.k);

    let file = File::open(path_to_iris_codes.as_path()).unwrap();
    let reader = BufReader::new(file);

    let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let mut irises: Vec<IrisCode> = Vec::with_capacity(args.num_irises);

    let (limit, skip, step) = match args.irises_selection {
        IrisSelection::All => (args.num_irises, 0, 1),
        IrisSelection::Even => (args.num_irises * 2, 0, 2),
        IrisSelection::Odd => (args.num_irises * 2, 1, 2),
    };

    let stream_iterator = stream
        .take(limit)
        .skip(skip)
        .step_by(step)
        .map(|json_pt| (&json_pt.unwrap()).into());

    irises.extend(stream_iterator);
    assert!(irises.len() == args.num_irises);

    let num_irises = irises.len();
    let mut engine = Engine::init(
        args.engine_choice,
        irises,
        args.k,
        num_already_processed + 1,
        args.num_threads,
    );

    let chunk_size = 2000;
    let mut evaluated_pairs = 0usize;
    println!("Starting work at serial id: {}", engine.next_id());

    let start_t = Instant::now();
    while engine.next_id() <= num_irises {
        let start = engine.next_id();
        let results = engine.compute_chunk(chunk_size);
        let end = engine.next_id();

        evaluated_pairs += (end - start) * num_irises;

        let mut file = OpenOptions::new()
            .append(true)
            .open(&args.results_file)
            .expect("Unable to open results file for appending");

        println!("Appending iris results from {} to {}", start, end - 1);
        for result in &results {
            let json_line = serde_json::to_string(result).expect("Failed to serialize KNNResult");
            writeln!(file, "{}", json_line).expect("Failed to write to results file");
        }
    }
    let duration = start_t.elapsed();
    println!(
        "naive_knn took {:?} (per evaluated pair)",
        duration.into_f64() / (evaluated_pairs as f64)
    );
}
