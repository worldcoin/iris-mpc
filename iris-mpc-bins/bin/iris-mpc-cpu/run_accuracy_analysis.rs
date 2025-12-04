use clap::Parser;
use eyre::Result;
use iris_mpc_cpu::analysis::accuracy::{
    load_graph, load_iris_store, process_results, run_analysis, Config,
};
use iris_mpc_cpu::utils::serialization::load_toml;
use metrics::histogram;
use metrics_tracing_context::{MetricsLayer, TracingContextLayer};
use metrics_util::debugging::{DebuggingRecorder, Snapshotter};
use metrics_util::layers::Layer;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
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
        .with(MetricsLayer::new()) // This is the missing piece
        .init();

    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();

    let recorder = TracingContextLayer::only_allow(&[
        "query_id", "mutation",
        "rotation",
        // "currently_visited",
        // "computed_ins_rate",
        // "target_batch_size",
        // "layer",
    ])
    .layer(recorder);
    metrics::set_global_recorder(recorder).expect("failed to install recorder");

    println!("Starting analysis...");
    let results = run_analysis(&config.analysis, store, graph, &mut rng).await?;

    process_results(&config.analysis, results)?;

    export_metrics_csv(&snapshotter, Path::new("metrics.csv"))?;

    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    Ok(())
}

#[derive(Serialize)]
struct MetricEntry {
    name: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    labels: HashMap<String, String>,
    value: MetricValue,
}

#[derive(Serialize)]
#[serde(untagged)]
enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram {
        avg: f64,
        n: usize,
        values: Vec<f64>,
    },
}

#[derive(Serialize)]
struct ContextGroup {
    query_id: String,
    mutation: String,
    rotation: String,
    metrics: Vec<MetricEntry>,
}

fn export_metrics_json(snapshotter: &Snapshotter, path: &Path) -> Result<()> {
    let snapshot = snapshotter.snapshot();
    let records: Vec<_> = snapshot.into_vec();

    let mut by_context: HashMap<(String, String, String), Vec<MetricEntry>> = HashMap::new();

    for (key, _, _, value) in records {
        let labels: HashMap<&str, &str> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();

        let context = (
            labels.get("query_id").unwrap_or(&"").to_string(),
            labels.get("mutation").unwrap_or(&"").to_string(),
            labels.get("rotation").unwrap_or(&"").to_string(),
        );

        let extra_labels: HashMap<String, String> = key
            .key()
            .labels()
            .filter(|l| !matches!(l.key(), "query_id" | "mutation" | "rotation"))
            .map(|l| (l.key().to_string(), l.value().to_string()))
            .collect();

        let metric_value = match value {
            metrics_util::debugging::DebugValue::Counter(v) => MetricValue::Counter(v),
            metrics_util::debugging::DebugValue::Gauge(v) => MetricValue::Gauge(v.into()),
            metrics_util::debugging::DebugValue::Histogram(v) => {
                let values: Vec<f64> = v.iter().map(|f| f64::from(*f)).collect();
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                MetricValue::Histogram {
                    avg,
                    n: values.len(),
                    values,
                }
            }
        };

        by_context.entry(context).or_default().push(MetricEntry {
            name: key.key().name().to_string(),
            labels: extra_labels,
            value: metric_value,
        });
    }

    let mut groups: Vec<ContextGroup> = by_context
        .into_iter()
        .map(|((query_id, mutation, rotation), metrics)| ContextGroup {
            query_id,
            mutation,
            rotation,
            metrics,
        })
        .collect();

    // Sort for deterministic output
    groups.sort_by(|a, b| {
        (&a.query_id, &a.mutation, &a.rotation).cmp(&(&b.query_id, &b.mutation, &b.rotation))
    });

    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &groups)?;

    Ok(())
}

#[derive(Debug, Default)]
struct MetricGroup {
    counters: HashMap<String, u64>,
    histograms: HashMap<String, Vec<f64>>,
}

fn export_metrics_grouped(snapshotter: &Snapshotter, path: &Path) -> Result<()> {
    use std::io::Write;

    let snapshot = snapshotter.snapshot();
    let records: Vec<_> = snapshot.into_vec();

    // Group by metric name
    let mut by_metric: HashMap<String, Vec<_>> = HashMap::new();
    for (key, unit, desc, value) in records {
        by_metric
            .entry(key.key().name().to_string())
            .or_default()
            .push((key, unit, desc, value));
    }

    let mut file = std::fs::File::create(path)?;

    for (metric_name, entries) in by_metric {
        // Find all label keys for this metric
        let mut label_keys: Vec<String> = entries
            .iter()
            .flat_map(|(key, _, _, _)| key.key().labels().map(|l| l.key().to_string()))
            .collect();
        label_keys.sort();
        label_keys.dedup();

        // Write section header
        writeln!(file, "\n=== {} ===", metric_name)?;

        // Write column header
        let mut header = label_keys.clone();
        header.push("value".to_string());
        writeln!(file, "{}", header.join("\t"))?;

        // Write rows
        for (key, _, _, value) in &entries {
            let labels: HashMap<&str, &str> =
                key.key().labels().map(|l| (l.key(), l.value())).collect();

            let value_str = match value {
                metrics_util::debugging::DebugValue::Counter(v) => v.to_string(),
                metrics_util::debugging::DebugValue::Histogram(v) => {
                    let avg: f64 = v.iter().map(|f| f64::from(*f)).sum::<f64>() / v.len() as f64;
                    format!("{:.2} (n={})", avg, v.len())
                }
                metrics_util::debugging::DebugValue::Gauge(v) => f64::from(*v).to_string(),
            };

            let mut row: Vec<String> = label_keys
                .iter()
                .map(|k| labels.get(k.as_str()).unwrap_or(&"").to_string())
                .collect();
            row.push(value_str);

            writeln!(file, "{}", row.join("\t"))?;
        }
    }

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

// fn export_metrics_csv(snapshotter: &Snapshotter, path: &Path) -> Result<()> {
//     let mut wtr = csv::Writer::from_path(path)?;

//     wtr.write_record(&["metric", "query_id", "mutation", "rotation", "value"])?;

//     let snapshot = snapshotter.snapshot();
//     for (key, _unit, _description, value) in snapshot.into_vec() {
//         let labels: HashMap<&str, &str> =
//             key.key().labels().map(|l| (l.key(), l.value())).collect();

//         let value_str = match value {
//             metrics_util::debugging::DebugValue::Counter(v) => v.to_string(),
//             metrics_util::debugging::DebugValue::Histogram(v) => v
//                 .iter()
//                 .map(|x| x.to_string())
//                 .collect::<Vec<_>>()
//                 .join(";"),
//             metrics_util::debugging::DebugValue::Gauge(v) => v.to_string(),
//         };

//         wtr.write_record(&[
//             key.key().name(),
//             labels.get("query_id").unwrap_or(&""),
//             labels.get("mutation").unwrap_or(&""),
//             labels.get("rotation").unwrap_or(&""),
//             &value_str,
//         ])?;
//     }

//     wtr.flush()?;
//     Ok(())
// }

fn organize_metrics(snapshotter: &Snapshotter) -> HashMap<(String, String, String), MetricGroup> {
    let mut grouped: HashMap<(String, String, String), MetricGroup> = HashMap::new();

    let snapshot = snapshotter.snapshot();
    for (key, _unit, _description, value) in snapshot.into_vec() {
        // Extract labels
        let labels: HashMap<&str, &str> =
            key.key().labels().map(|l| (l.key(), l.value())).collect();

        let query_id = labels.get("query_id").unwrap_or(&"").to_string();
        let mutation = labels.get("mutation").unwrap_or(&"").to_string();
        let rotation = labels.get("rotation").unwrap_or(&"").to_string();

        let group_key = (query_id, mutation, rotation);
        let group = grouped.entry(group_key).or_default();

        let metric_name = key.key().name().to_string();

        match value {
            metrics_util::debugging::DebugValue::Counter(v) => {
                group.counters.insert(metric_name, v);
            }
            metrics_util::debugging::DebugValue::Histogram(v) => {
                group.histograms.insert(
                    metric_name,
                    v.iter().map(|f| (*f).into()).collect::<Vec<_>>(),
                );
            }
            metrics_util::debugging::DebugValue::Gauge(v) => {
                // Handle if needed
            }
        }
    }

    grouped
}

fn print_organized_metrics(snapshotter: &Snapshotter) {
    let grouped = organize_metrics(snapshotter);

    for ((query_id, mutation, rotation), group) in &grouped {
        println!(
            "\n=== query_id={}, mutation={}, rotation={} ===",
            query_id, mutation, rotation
        );

        for (name, value) in &group.counters {
            println!("  {}: {}", name, value);
        }

        for (name, values) in &group.histograms {
            let avg: f64 = values.iter().sum::<f64>() / values.len() as f64;
            println!("  {}: avg={:.2}, n={}", name, avg, values.len());
        }
    }
}

fn print_metrics(snapshotter: &Snapshotter) {
    let snapshot = snapshotter.snapshot();
    for (key, _unit, _description, value) in snapshot.into_vec() {
        println!("{:?}: {:?}", key.key(), value);
    }
}
