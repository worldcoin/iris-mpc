use clap::Parser;
use eyre::Result;
use iris_mpc_cpu::analysis::accuracy::{
    load_graph, load_iris_store, process_results, run_analysis, Config,
};
use iris_mpc_cpu::utils::serialization::load_toml;
use std::path::PathBuf;

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
    let mut rng = rand::SeedableRng::seed_from_u64(config.analysis.seed.unwrap_or(0));
    //TODO: (?)separate analysis rng from iris and graph rng

    let mut store =
        load_iris_store(config.irises, &mut rng, config.analysis.get_distance_fn()?).await?;
    println!(
        "Loaded {} iris codes into PlaintextStore with distance_fn = {:?}.",
        store.len(),
        store.distance_fn
    );

    println!("Initializing graph...");
    let graph = load_graph(&config.graph, &mut store, &mut rng).await?;
    println!("Graph initialized.");

    println!("Starting analysis...");
    let results = run_analysis(config.analysis.clone(), store, graph, &mut rng).await?;
    println!("Analysis complete. {} searches performed.", results.len());

    process_results(&config.analysis, results)?;
    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    Ok(())
}
