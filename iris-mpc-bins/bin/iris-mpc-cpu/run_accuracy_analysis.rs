use clap::Parser;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_common::IrisVectorId as VectorId;
use iris_mpc_cpu::analysis::accuracy::{
    load_graph, load_iris_store, process_results, run_analysis, AnalysisConfig, Config, IrisesInit,
};
use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
use iris_mpc_cpu::hnsw::{GraphMem, HnswSearcher};
use iris_mpc_cpu::utils::serialization::graph::{read_graph_from_file, GraphFormat};
use iris_mpc_cpu::utils::serialization::iris_ndjson::IrisSelection;
use iris_mpc_cpu::utils::serialization::load_toml;
use rand::{rngs::StdRng, SeedableRng};
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config: Config = load_toml(&cli.job_spec)?;

    println!("Configuration loaded from {}.", cli.job_spec.display());

    let (mut store, mut rng) = load_iris_store(&config.irises, config.analysis.seed).await?;
    println!("Loaded {} iris codes into PlaintextStore.", store.len());

    store.distance_fn = config.analysis.get_distance_fn()?;

    let (graph, searcher) = load_graph(&config.graph, &mut store, &mut rng).await?;
    println!("Graph initialized.");

    println!("Starting analysis...");
    let results = run_analysis(&config.analysis, store, graph, &searcher, &mut rng).await?;
    println!("Analysis complete. {} searches performed.", results.len());

    process_results(&config.analysis, results)?;
    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    Ok(())
}
