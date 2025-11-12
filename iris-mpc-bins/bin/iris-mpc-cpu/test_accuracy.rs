use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::IrisVectorId as VectorId;
use iris_mpc_cpu::analysis::accuracy::{process_results, run_analysis, AnalysisConfig};
use iris_mpc_cpu::hawkers::aby3::aby3_store::DistanceFn;
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::{GraphMem, HnswSearcher};
use iris_mpc_cpu::utils::serialization::graph::{read_graph_from_file, GraphFormat};
use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;
use iris_mpc_cpu::utils::serialization::load_toml;
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio;

// --- CLI and Configuration ---

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the configuration TOML file
    #[clap(long)]
    job_spec: PathBuf,
}

/// Top-level configuration struct, loaded from TOML.
#[derive(Debug, Deserialize)]
struct Config {
    irises: IrisesInit,
    graph: GraphInit,
    analysis: AnalysisConfig,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "option")]
enum IrisesInit {
    /// Load iris codes from a .ndjson file.
    NdjsonFile {
        path: PathBuf,
        limit: Option<usize>,
        selection: Option<IrisSelection>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "option")]
enum GraphInit {
    /// Load a pre-built graph from a binary file.
    BinFile { path: PathBuf, format: GraphFormat },
}

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI and Config
    let cli = Cli::parse();
    let config: Config = load_toml(&cli.job_spec)?;

    println!("Configuration loaded from {}.", cli.job_spec.display());

    // 2. Initialize Iris Codes and PlaintextStore
    let (mut store, mut rng) = load_iris_store(&config.irises, config.analysis.seed).await?;
    println!("Loaded {} iris codes into PlaintextStore.", store.len());

    // 4. Set Analysis Parameters on Store
    store.distance_fn = match config.analysis.distance_fn.as_str() {
        "fhd" => DistanceFn::FHD,
        "min_fhd" => DistanceFn::MinFHD,
        _ => bail!("Unknown distance_fn: {}", config.analysis.distance_fn),
    };

    // 3. Initialize Graph
    let (graph, searcher) = load_graph(&config.graph).await?;
    println!("Graph initialized.");

    // 5. Execute Analysis
    println!("Starting analysis...");
    let results = run_analysis(&config.analysis, store, graph, &searcher, &mut rng).await?;
    println!("Analysis complete. {} searches performed.", results.len());

    process_results(&config.analysis, results)?;
    //dbg!(&results);
    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    Ok(())
}

// --- Helper Functions ---

/// Loads iris codes into a `PlaintextStore` based on `IrisesInit` config.
/// Returns the store and an RNG for use in later steps.
async fn load_iris_store(
    config: &IrisesInit,
    seed: Option<u64>,
) -> Result<(PlaintextStore, StdRng)> {
    // This RNG is for graph building and analysis, seeded by the *analysis* seed.
    let rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

    let store = match config {
        IrisesInit::NdjsonFile {
            path,
            limit,
            selection,
        } => {
            let mut path = path.clone();
            if path.is_relative() {
                if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
                    path = Path::new(&manifest_dir).join(path);
                }
            }
            println!("Loading irises from NDJSON file: {}", path.display());
            PlaintextStore::from_ndjson_file(
                &path,
                *limit,
                selection.unwrap_or(IrisSelection::All),
            )?
        }
    };

    Ok((store, rng))
}

/// Loads or builds the HNSW graph.
async fn load_graph(config: &GraphInit) -> Result<(GraphMem<VectorId>, HnswSearcher)> {
    match config {
        GraphInit::BinFile { path, format } => {
            println!("Loading graph from binary file: {}", path.display());
            let graph = read_graph_from_file(path, *format)?;
            let searcher = HnswSearcher::new_with_test_parameters();
            Ok((graph, searcher))
        }
    }
}
