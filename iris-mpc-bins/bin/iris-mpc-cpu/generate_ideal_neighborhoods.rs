use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

use clap::Parser;
use iris_mpc_common::{iris_db::iris::IrisCode, vector_id::SerialId};
use iris_mpc_cpu::{
    hawkers::ideal_knn_engines::{Engine, EngineInt4, EngineKind, KNNResult},
    utils::serialization::{
        int4_ndjson::int4_vectors_from_ndjson,
        iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
    },
};
use metrics::IntoF64;

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A struct to hold the metadata stored in the first line of the results file.
#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct ResultsHeader {
    /// Iris selection used to load the input. `Some(..)` for iris engines,
    /// `None` for the deep-ID Int4 engine (the Int4 loader has no selection).
    iris_selection: Option<IrisSelection>,
    /// Engine identity, so resuming a file rejects a mismatched distance/store.
    engine_choice: EngineKind,
    num_vectors: usize,
    k: usize,
}

/// Dispatch wrapper so the chunk/append loop is written once regardless of
/// store kind. Both inner engines return `Vec<KNNResult>`.
enum AnyEngine {
    Iris(Engine),
    Int4(EngineInt4),
}

impl AnyEngine {
    fn next_id(&self) -> SerialId {
        match self {
            AnyEngine::Iris(e) => e.next_id(),
            AnyEngine::Int4(e) => e.next_id(),
        }
    }

    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        match self {
            AnyEngine::Iris(e) => e.compute_chunk(chunk_size),
            AnyEngine::Int4(e) => e.compute_chunk(chunk_size),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of irises to process
    #[arg(long, alias = "num_irises", default_value_t = 1000)]
    num_vectors: usize,

    /// Path to the input file of vectors: iris codes ndjson for iris engines,
    /// or Int4 vectors ndjson when --engine-choice is NaiveInt4Dot.
    #[arg(
        long = "input",
        alias = "path-to-iris-codes",
        default_value = "data/store.ndjson"
    )]
    path_to_vectors: PathBuf,

    /// The k for k-NN
    #[arg(long)]
    k: usize,

    /// Path to the results file
    #[arg(long)]
    results_file: PathBuf,

    /// Selection of irises to process (iris engines only; ignored for NaiveInt4Dot).
    #[arg(long, value_enum, default_value_t = IrisSelection::All)]
    irises_selection: IrisSelection,

    /// Engine (distance + store kind). Iris variants use --irises-selection;
    /// NaiveInt4Dot reads a deep-ID Int4 ndjson and ignores --irises-selection.
    #[arg(long, value_enum, default_value_t = EngineKind::NaiveFHD)]
    engine_choice: EngineKind,
}
#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Iris selection only applies to iris engines; the Int4 loader has none.
    let header_selection = if args.engine_choice.is_int4() {
        None
    } else {
        Some(args.irises_selection)
    };

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
                iris_selection: header_selection,
                engine_choice: args.engine_choice,
                num_vectors: args.num_vectors,
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

            let nodes: Vec<SerialId> = deserialized_results
                .into_iter()
                .map(|result| result.node.serial_id())
                .collect();
            (nodes.len() as SerialId, nodes)
        }
        Err(_) => {
            // File doesn't exist, create it and write the header.
            let mut file = File::create(&args.results_file).expect("Unable to create results file");
            let header = ResultsHeader {
                iris_selection: header_selection,
                engine_choice: args.engine_choice,
                num_vectors: args.num_vectors,
                k: args.k,
            };
            let header_str =
                serde_json::to_string(&header).expect("Failed to serialize ResultsHeader");
            writeln!(file, "{}", header_str).expect("Failed to write header to new results file");
            (0, Vec::new())
        }
    };

    if num_already_processed > 0 {
        let expected_nodes: Vec<SerialId> = (1..num_already_processed + 1).collect();
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

    let path_to_vectors = args.path_to_vectors;
    assert!(args.num_vectors > args.k);

    let next_id = num_already_processed + 1;
    let (num_vectors, mut engine): (SerialId, AnyEngine) = if args.engine_choice.is_int4() {
        let echoice = args
            .engine_choice
            .as_int4()
            .expect("is_int4() implies as_int4() is Some");
        let vectors =
            match int4_vectors_from_ndjson(path_to_vectors.as_path(), Some(args.num_vectors)) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error: Failed to load Int4 vectors input file.");
                    eprintln!(" -> Error details: {}", e);
                    std::process::exit(1);
                }
            };
        assert_eq!(vectors.len(), args.num_vectors);
        let n = vectors.len() as SerialId;
        (
            n,
            AnyEngine::Int4(EngineInt4::init(echoice, vectors, args.k, next_id)),
        )
    } else {
        let echoice = args
            .engine_choice
            .as_iris()
            .expect("non-int4 implies as_iris() is Some");
        let mut irises: Vec<IrisCode> = Vec::with_capacity(args.num_vectors);
        let stream_iterator = match irises_from_ndjson_iter(
            path_to_vectors.as_path(),
            Some(args.num_vectors),
            args.irises_selection,
        ) {
            Ok(iter) => iter,
            Err(e) => {
                eprintln!("Error: Failed to open irises input file.");
                eprintln!(" -> Error details: {}", e);
                std::process::exit(1);
            }
        };
        irises.extend(stream_iterator);
        assert_eq!(irises.len(), args.num_vectors);
        let n = irises.len() as SerialId;
        (
            n,
            AnyEngine::Iris(Engine::init(echoice, irises, args.k, next_id)),
        )
    };

    let chunk_size = 2000;
    let mut evaluated_pairs = 0u64;
    println!("Starting work at serial id: {}", engine.next_id());

    let start_t = Instant::now();
    while engine.next_id() <= num_vectors {
        let start = engine.next_id();
        let results = engine.compute_chunk(chunk_size);
        let end = engine.next_id();

        evaluated_pairs += ((end - start) as u64) * (num_vectors as u64);

        let mut file = OpenOptions::new()
            .append(true)
            .open(&args.results_file)
            .expect("Unable to open results file for appending");

        println!("Appending results from {} to {}", start, end - 1);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_iris() {
        let header = ResultsHeader {
            iris_selection: Some(IrisSelection::Odd),
            engine_choice: EngineKind::NaiveFHD,
            num_vectors: 1000,
            k: 16,
        };
        let s = serde_json::to_string(&header).unwrap();
        let back: ResultsHeader = serde_json::from_str(&s).unwrap();
        assert_eq!(header, back);
        assert_eq!(back.iris_selection, Some(IrisSelection::Odd));
        assert!(!back.engine_choice.is_int4());
    }

    #[test]
    fn header_roundtrip_int4() {
        let header = ResultsHeader {
            iris_selection: None,
            engine_choice: EngineKind::NaiveInt4Dot,
            num_vectors: 4096,
            k: 32,
        };
        let s = serde_json::to_string(&header).unwrap();
        let back: ResultsHeader = serde_json::from_str(&s).unwrap();
        assert_eq!(header, back);
        assert_eq!(back.iris_selection, None);
        assert!(back.engine_choice.is_int4());
    }
}
