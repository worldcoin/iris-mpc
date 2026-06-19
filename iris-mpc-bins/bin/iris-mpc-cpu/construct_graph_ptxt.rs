use std::{iter::once, path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::{bail, Result};
use iris_mpc_common::{iris_db::iris::IrisCode, VectorId};
use itertools::{chain, Itertools};
use serde::Deserialize;

use iris_mpc_cpu::{
    hawkers::{
        aby3::aby3_store::{DistanceMode, DistanceOps, FhdOps, NhdOps},
        build_plaintext::{deep_id_parallel_batch_insert, plaintext_parallel_batch_insert},
        plaintext_deep_id_store::{Int4Vector, PlaintextDeepIDStore, SharedPlaintextDeepIDStore},
        plaintext_store::SharedPlaintextStore,
    },
    hnsw::{vector_store::VectorStoreMut, GraphMem, HnswSearcher},
    utils::{
        cli::{Int4VectorsConfig, IrisesConfig, LoadGraphConfig, SearcherConfig},
        serialization::{graph::write_graph_to_file, load_store_kind_toml},
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

/// Selects the distance operations type at runtime.
#[derive(Clone, Copy, Debug, Deserialize, Default)]
enum DistanceOpsKind {
    #[default]
    Fhd,
    Nhd,
}

#[derive(Clone, Debug, Deserialize)]
struct CliConfig {
    /// Store variant: legacy iris codes or new deep-ID Int4 vectors. The
    /// variant is selected by which fields are present at the top level.
    #[serde(flatten)]
    store: StoreKindConfig,

    /// Path and version info for loading an initial graph from file.
    graph: Option<LoadGraphConfig>,

    /// Configuration to specify HnswSearcher struct for graph construction.
    searcher: SearcherConfig,

    /// Seed value used for sampling graph layers for insertion.
    hnsw_prf_seed: Option<u64>,

    /// Specifies the method of outputing graph data to file after construction.
    output: OutputGraphConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StoreKindConfig {
    Iris {
        /// Specification for iris codes to use in the construction.
        irises: IrisesConfig,

        /// Distance function for comparison of iris codes.
        distance_fn: DistanceMode,

        /// Selects the distance operations type (Fhd or Nhd). Defaults to Fhd.
        #[serde(default)]
        distance_ops: DistanceOpsKind,
    },
    DeepID {
        /// Specification for deep-ID Int4 vectors to use in the construction.
        vectors: Int4VectorsConfig,

        /// Inner-product match threshold (used to construct the store; not
        /// directly consulted during graph build).
        threshold: i32,
    },
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

/// Load an existing graph from `graph_spec` (or initialize a fresh one) and
/// return it along with its highest layer-0 node serial id.
fn load_existing_graph(graph_spec: Option<LoadGraphConfig>) -> Result<(GraphMem, u32)> {
    if let Some(graph_spec) = graph_spec {
        tracing::info!("Loading graph from file");
        let graph = graph_spec.read_graph_from_file()?;
        let graph_max_id = graph
            .get_layers()
            .first()
            .map(|layer_0| layer_0.links.keys().copied().max().unwrap_or(0))
            .unwrap_or(0);
        tracing::info!("Loaded graph has max node id {graph_max_id}");
        Ok((graph, graph_max_id))
    } else {
        tracing::info!("Initializing graph");
        Ok((GraphMem::new(), 0))
    }
}

#[allow(non_snake_case)]
async fn build_iris_graph<D: DistanceOps>(
    irises: Vec<(VectorId, IrisCode)>,
    graph_spec: Option<LoadGraphConfig>,
    distance_fn: DistanceMode,
    searcher: &HnswSearcher,
    prf_seed: &[u8; 16],
    output: OutputGraphConfig,
) -> Result<()> {
    let (mut graph, graph_max_id) = load_existing_graph(graph_spec)?;

    let start_idx = graph_max_id as usize;
    let end_idx = irises.len();

    if start_idx >= end_idx {
        bail!(
            "Graph max ID ({}) exceeds available iris codes ({})",
            start_idx,
            end_idx
        );
    }

    let first_id = irises[start_idx].0.serial_id();
    let last_id = irises[end_idx - 1].0.serial_id();

    // insert irises which are already represented in the graph
    tracing::info!("Initializing vector store");
    let mut store = SharedPlaintextStore::<D>::new();
    store.distance_mode = distance_fn;
    for (id, iris) in irises[0..(graph_max_id as usize)].iter() {
        let _id = store.insert_at(id, &Arc::new(iris.clone())).await?;
    }

    match output {
        OutputGraphConfig::Simple { path } => {
            tracing::info!("Building HNSW graph for ids {first_id} to {last_id}");
            let new_irises = irises[(graph_max_id as usize)..].to_vec();
            (graph, _) =
                plaintext_parallel_batch_insert(graph, store, new_irises, searcher, 1, prf_seed)
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
                    graph,
                    store,
                    i_new_irises,
                    searcher,
                    1,
                    prf_seed,
                )
                .await?;

                let filename = format!("{filename_stem}_{i_end}.dat");
                let output_path = base_directory.join(filename.clone());

                tracing::info!("Persisting HNSW graph to file: {filename}");
                write_graph_to_file(output_path, graph.clone())?;
            }
        }
    }

    Ok(())
}

async fn build_deep_id_graph(
    vectors: Vec<(VectorId, Int4Vector)>,
    threshold: i32,
    graph_spec: Option<LoadGraphConfig>,
    searcher: &HnswSearcher,
    prf_seed: &[u8; 16],
    output: OutputGraphConfig,
) -> Result<()> {
    let (mut graph, graph_max_id) = load_existing_graph(graph_spec)?;

    let start_idx = graph_max_id as usize;
    let end_idx = vectors.len();
    if start_idx >= end_idx {
        bail!(
            "Graph max ID ({}) exceeds available deep-ID vectors ({})",
            start_idx,
            end_idx
        );
    }

    let first_id = vectors[start_idx].0.serial_id();
    let last_id = vectors[end_idx - 1].0.serial_id();

    tracing::info!("Initializing deep-ID vector store");
    let mut store_ = PlaintextDeepIDStore::new(threshold);
    for (id, v) in vectors[0..start_idx].iter() {
        store_.insert_with_id(*id, Arc::new(v.clone()));
    }
    let mut store: SharedPlaintextDeepIDStore = store_.into();

    match output {
        OutputGraphConfig::Simple { path } => {
            tracing::info!("Building deep-ID HNSW graph for ids {first_id} to {last_id}");
            let new_vectors = vectors[start_idx..].to_vec();
            (graph, _) =
                deep_id_parallel_batch_insert(graph, store, new_vectors, searcher, 1, prf_seed)
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

            tracing::info!(
                "Building deep-ID HNSW graph with checkpoints, for ids {first_id} to {last_id}"
            );
            for (checkpoint_idx, (i_start, i_end)) in intervals.iter().enumerate() {
                tracing::info!(
                    "Planned checkpoint {}: ids {} to {}",
                    checkpoint_idx + 1,
                    vectors[*i_start].0.serial_id(),
                    vectors[*i_end - 1].0.serial_id(),
                );
            }

            for (checkpoint_idx, (i_start, i_end)) in intervals.into_iter().enumerate() {
                tracing::info!(
                    "Building deep-ID graph checkpoint {} for ids {} to {}...",
                    checkpoint_idx + 1,
                    vectors[i_start].0.serial_id(),
                    vectors[i_end - 1].0.serial_id(),
                );
                let chunk = vectors[i_start..i_end].to_vec();
                (graph, store) =
                    deep_id_parallel_batch_insert(graph, store, chunk, searcher, 1, prf_seed)
                        .await?;

                let filename = format!("{filename_stem}_{i_end}.dat");
                let output_path = base_directory.join(filename.clone());
                tracing::info!("Persisting HNSW graph to file: {filename}");
                write_graph_to_file(output_path, graph.clone())?;
            }
        }
    }

    Ok(())
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
    let config: CliConfig = load_store_kind_toml(&cli.job_spec)?;
    tracing::info!("Configuration loaded from {}", cli.job_spec.display());

    tracing::info!("Initializing searcher");
    let searcher: HnswSearcher = (&config.searcher).try_into()?;
    let prf_seed = (config.hnsw_prf_seed.unwrap_or(0) as u128).to_le_bytes();

    match config.store.clone() {
        StoreKindConfig::Iris {
            irises,
            distance_fn,
            distance_ops,
        } => {
            tracing::info!("Loading iris codes");
            let irises_ = iris_mpc_cpu::utils::cli::load_irises(irises).await?;
            let irises: Vec<_> = irises_
                .into_iter()
                .enumerate()
                .map(|(idx, code)| (VectorId::from_0_index(idx as u32), code))
                .collect();
            if irises.is_empty() {
                bail!("Iris DB is empty");
            }
            tracing::info!("Loaded {} iris codes to memory", irises.len());

            tracing::info!("Using distance ops: {:?}", distance_ops);
            match distance_ops {
                DistanceOpsKind::Fhd => {
                    build_iris_graph::<FhdOps>(
                        irises,
                        config.graph,
                        distance_fn,
                        &searcher,
                        &prf_seed,
                        config.output,
                    )
                    .await?;
                }
                DistanceOpsKind::Nhd => {
                    build_iris_graph::<NhdOps>(
                        irises,
                        config.graph,
                        distance_fn,
                        &searcher,
                        &prf_seed,
                        config.output,
                    )
                    .await?;
                }
            }
        }
        StoreKindConfig::DeepID { vectors, threshold } => {
            tracing::info!("Loading deep-ID vectors");
            let vectors_ = iris_mpc_cpu::utils::cli::load_int4_vectors(vectors).await?;
            let vectors: Vec<_> = vectors_
                .into_iter()
                .enumerate()
                .map(|(idx, v)| (VectorId::from_0_index(idx as u32), v))
                .collect();
            if vectors.is_empty() {
                bail!("Deep-ID DB is empty");
            }
            tracing::info!("Loaded {} deep-ID vectors to memory", vectors.len());

            build_deep_id_graph(
                vectors,
                threshold,
                config.graph,
                &searcher,
                &prf_seed,
                config.output,
            )
            .await?;
        }
    }

    tracing::info!("Done!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_iris_config_deserializes_as_iris_variant() {
        let toml_str = r#"
distance_fn = "MinRotation"
hnsw_prf_seed = 42

[irises]
option = "Random"
number = 64
seed = 1

[searcher]
max_graph_layer = 1

[searcher.params]
option = "Standard"
ef_constr = 32
ef_search = 32
M = 16

[output]
path = "data/out.dat"
"#;
        let cfg: CliConfig = toml::from_str(toml_str).expect("legacy iris TOML deserializes");
        match cfg.store {
            StoreKindConfig::Iris { .. } => {}
            other => panic!("expected Iris, got {other:?}"),
        }
    }

    #[test]
    fn deepid_config_deserializes_as_deepid_variant() {
        let toml_str = r#"
hnsw_prf_seed = 42
threshold = 5000

[vectors]
option = "Random"
number = 64
seed = 1

[searcher]
max_graph_layer = 1

[searcher.params]
option = "Standard"
ef_constr = 32
ef_search = 32
M = 16

[output]
path = "data/out.dat"
"#;
        let cfg: CliConfig = toml::from_str(toml_str).expect("deepid TOML deserializes");
        match cfg.store {
            StoreKindConfig::DeepID { threshold, .. } => assert_eq!(threshold, 5000),
            other => panic!("expected DeepID, got {other:?}"),
        }
    }
}
