use clap::Parser;
use eyre::Result;
use iris_mpc_cpu::analysis::accuracy::{
    load_graph, load_iris_store, process_results, run_analysis, Config,
};
use iris_mpc_cpu::utils::serialization::load_toml;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use metrics_util::layers::Layer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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

    tracing_subscriber::registry()
        .with(MetricsLayer::new())
        .init();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let recorder = TracingContextLayer::only_allow(&["__query_id", "__mutation", "__rotation"])
        .layer(recorder);
    metrics::set_global_recorder(recorder).expect("failed to install recorder");

    println!("Starting analysis...");
    let results = run_analysis(&config.analysis, store, graph, &mut rng).await?;

    process_results(&config.analysis, results)?;

    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    export_metrics_csv(&snapshotter, Path::new("metrics.csv"))?;

    println!(
        "Metrics written to {}.",
        config.analysis.metrics_path.display()
    );

    Ok(())
}

fn export_metrics_csv(snapshotter: &Snapshotter, path: &Path) -> Result<()> {
    let snapshot = snapshotter.snapshot();
    let records: Vec<_> = snapshot.into_vec();

    // Collect all unique label keys across all metrics
    let mut all_label_keys: Vec<String> = records
        .iter()
        .flat_map(|(key, _, _, _)| key.key().labels().map(|l| l.key().to_string()))
        .collect();
    all_label_keys.sort();
    all_label_keys.dedup();

    let mut wtr = csv::Writer::from_path(path)?;

    // Header: metric, <all label keys>, value
    let mut header = vec!["metric".to_string()];
    header.extend(all_label_keys.clone());
    header.push("value".to_string());
    wtr.write_record(&header)?;

    // Write each metric
    for (key, _unit, _description, value) in records {
        let labels: HashMap<&str, &str> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();

        let value_str = match value {
            metrics_util::debugging::DebugValue::Counter(v) => v.to_string(),
            metrics_util::debugging::DebugValue::Histogram(v) => v
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(";"),
            metrics_util::debugging::DebugValue::Gauge(v) => v.to_string(),
        };

        let mut row = vec![key.key().name().to_string()];
        for label_key in &all_label_keys {
            row.push(labels.get(label_key.as_str()).unwrap_or(&"").to_string());
        }
        row.push(value_str);

        wtr.write_record(&row)?;
    }

    wtr.flush()?;
    Ok(())
}
