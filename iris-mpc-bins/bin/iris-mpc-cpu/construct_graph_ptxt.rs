use std::{iter::once, path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::IrisVectorId;
use itertools::{chain, Itertools};
use serde::Deserialize;

use iris_mpc_cpu::{
    hawkers::{
        aby3::aby3_store::DistanceFn, build_plaintext::plaintext_parallel_batch_insert,
        plaintext_store::SharedPlaintextStore,
    },
    hnsw::{vector_store::VectorStoreMut, GraphMem, HnswSearcher},
    utils::{
        cli::{IrisesConfig, LoadGraphConfig, SearcherConfig},
        serialization::{graph::write_graph_to_file, load_toml},
    },
};
use tracing_subscriber::{prelude::*, EnvFilter};

#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    /// Path to configuration TOML file.
    #[clap(long)]
    job_spec: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct CliConfig {
    /// Specification for iris codes to use in the construction.
    irises: IrisesConfig,

    /// Path and version info for loading an initial graph from file.
    graph: Option<LoadGraphConfig>,

    /// Configuration to specify HnswSearcher struct for graph construction.
    searcher: SearcherConfig,

    /// Distance function for comparison of iris codes.
    distance_fn: DistanceFn,

    /// Seed value used for sampling graph layers for insertion.
    hnsw_prf_seed: Option<u64>,

    /// Specifies the method of outputing graph data to file after construction.
    output: OutputGraphConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum OutputGraphConfig {
    Simple {
        path: PathBuf,
    },
    Checkpoints {
        base_directory: PathBuf,
        filename_stem: String,
        checkpoints: Checkpoints,
    },
}

#[derive(Clone, Debug, Deserialize)]
enum Checkpoints {
    Regular(usize),
    Values(Vec<usize>),
}

impl Checkpoints {
    /// Return intervals of indices splitting the provided range at the checkpoints
    pub fn intervals_for_range(&self, start: usize, end: usize) -> Vec<(usize, usize)> {
        if start > end {
            return Vec::new();
        }

        let intermediate = match self.clone() {
            Checkpoints::Regular(number) => {
                // compute multiples of `number` strictly between `start` and `end`
                let first = (start + 1).div_ceil(number);
                let last = end.div_ceil(number);
                if first >= last {
                    Vec::new()
                } else {
                    (first..last).map(|i| i * number).collect()
                }
            }
            Checkpoints::Values(mut values) => {
                values.sort();
                values
                    .into_iter()
                    .filter(|&val| val > start && val < end)
                    .collect()
            }
        };

        chain!(once(start), intermediate, once(end))
            .tuple_windows()
            .collect()
    }
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    let filter = EnvFilter::new("info,iris_mpc_cpu::hnsw=off");

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(filter))
        .init();

    // parse args
    tracing::info!("Loading configuration from file");
    let cli = Args::parse();
    let config: CliConfig = load_toml(&cli.job_spec)?;
    tracing::info!("Configuration loaded from {}", cli.job_spec.display());

    tracing::info!("Initializing searcher");
    let searcher: HnswSearcher = (&config.searcher).try_into()?;
    let prf_seed = (config.hnsw_prf_seed.unwrap_or(0) as u128).to_le_bytes();

    tracing::info!("Loading iris codes");
    let irises_ = iris_mpc_cpu::utils::cli::load_irises(config.irises).await?;
    let irises: Vec<_> = irises_
        .into_iter()
        .enumerate()
        .map(|(idx, code)| (IrisVectorId::from_0_index(idx as u32), code))
        .collect();
    tracing::info!("Loaded {} iris codes to memory", irises.len());

    let (mut graph, graph_max_id) = if let Some(graph_spec) = config.graph {
        tracing::info!("Loading graph from file");
        let graph = graph_spec.read_graph_from_file()?;
        let graph_max_id = graph
            .get_layers()
            .first()
            .map(|layer_0| {
                layer_0
                    .links
                    .keys()
                    .map(|id| id.serial_id())
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);
        tracing::info!("Loaded graph has max node id {graph_max_id}");
        (graph, graph_max_id)
    } else {
        tracing::info!("Initializing graph");
        (GraphMem::new(), 0)
    };

    let start_idx = graph_max_id as usize;
    let end_idx = irises.len();

    let first_id = irises[start_idx].0.serial_id();
    let last_id = irises[end_idx - 1].0.serial_id();

    // insert irises which are already represented in the graph
    tracing::info!("Initializing vector store");
    let mut store = SharedPlaintextStore::new();
    store.distance_fn = config.distance_fn;
    for (id, iris) in irises[0..(graph_max_id as usize)].iter() {
        let _id = store.insert_at(id, &Arc::new(iris.clone())).await?;
    }

    match config.output {
        OutputGraphConfig::Simple { path } => {
            tracing::info!("Building HNSW graph for ids {first_id} to {last_id}");
            let new_irises = irises[(graph_max_id as usize)..].to_vec();
            (graph, _) = plaintext_parallel_batch_insert(
                Some(graph),
                Some(store),
                new_irises,
                &searcher,
                1,
                &prf_seed,
            )
            .await?;

            tracing::info!("Persisting HNSW graph to file");
            write_graph_to_file(path, graph)?;
        }
        OutputGraphConfig::Checkpoints {
            base_directory,
            filename_stem,
            checkpoints,
        } => {
            if !base_directory.is_dir() {
                bail!("Specified base directory for graph outputs does not exist");
            }

            let intervals = checkpoints.intervals_for_range(start_idx, end_idx);

            tracing::info!("Building HNSW graph with checkpoints, for ids {first_id} to {last_id}");
            for (checkpoint_idx, (i_start, i_end)) in intervals.iter().enumerate() {
                tracing::info!(
                    "Planned checkpoint {}: ids {} to {}",
                    checkpoint_idx + 1,
                    irises[*i_start].0.serial_id(),
                    irises[*i_end - 1].0.serial_id(),
                )
            }

            for (checkpoint_idx, (i_start, i_end)) in intervals.into_iter().enumerate() {
                tracing::info!(
                    "Building graph checkpoint {} for ids {} to {}...",
                    checkpoint_idx + 1,
                    irises[i_start].0.serial_id(),
                    irises[i_end - 1].0.serial_id(),
                );
                let i_new_irises = irises[i_start..i_end].to_vec();
                (graph, store) = plaintext_parallel_batch_insert(
                    Some(graph),
                    Some(store),
                    i_new_irises,
                    &searcher,
                    1,
                    &prf_seed,
                )
                .await?;

                let filename = format!("{filename_stem}-{i_end}.dat");
                let output_path = base_directory.join(filename.clone());

                tracing::info!("Persisting HNSW graph to file: {filename}");
                write_graph_to_file(output_path, graph.clone())?;
            }
        }
    }

    tracing::info!("Done!");

    Ok(())
}
