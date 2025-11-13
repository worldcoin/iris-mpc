use crate::{
    hawkers::{
        aby3::aby3_store::DistanceFn,
        plaintext_store::{PlaintextStore, SharedPlaintextStore},
    },
    hnsw::{GraphMem, HnswSearcher},
    utils::serialization::{
        graph::{read_graph_from_file, GraphFormat},
        iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
    },
};
use eyre::{bail, eyre, Result};
use futures::future::JoinAll;
use iris_mpc_common::{
    iris_db::iris::{IrisCode, IrisMutationFamily},
    IrisSerialId, IrisVectorId as VectorId,
};
use itertools::{izip, Itertools};
use rand::{rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
    path::PathBuf,
    sync::Arc,
};
use tokio::task::JoinError;

/// Configuration for the accuracy analysis run.
#[derive(Debug, Deserialize)]
pub struct AnalysisConfig {
    /// Number of existing iris codes to sample for the test.
    pub sample_size: usize,
    /// Optional seed for reproducible sampling and mutation.
    pub seed: Option<u64>,
    /// ef_search parameter to use during analysis.
    pub ef_search: usize,
    /// Distance function: "fhd" (Fractional Hamming) or "min_fhd" (Min-FHD).
    pub distance_fn: String,
    /// Number of neighbors to retrieve in search (k).
    pub k_neighbors: usize,
    /// Output format: "rate" or "full_csv".
    pub output_format: String,
    /// Path for the output CSV file.
    pub output_path: PathBuf,
    /// Range of relative rotations to test (e.g., [-3, -2, -1, 0, 1, 2, 3]).
    pub rotations: Range<isize>,
    /// List of mutation amounts
    pub mutations: Vec<f64>,
}

impl AnalysisConfig {
    pub fn get_distance_fn(&self) -> Result<DistanceFn> {
        match self.distance_fn.as_str() {
            "fhd" => Ok(DistanceFn::FHD),
            "min_fhd" => Ok(DistanceFn::MinFHD),
            _ => bail!("Unknown distance_fn: {}", self.distance_fn),
        }
    }
}

/// Struct to hold a single search result for analysis.
#[derive(Debug, Serialize)]
pub struct AnalysisResult {
    id: IrisSerialId,
    mutation: f64,
    rotation: isize,
    found: bool,
}
pub async fn run_analysis(
    config: &AnalysisConfig,
    store: PlaintextStore,
    graph: GraphMem<VectorId>,
    searcher: &HnswSearcher,
    rng: &mut StdRng,
) -> Result<Vec<AnalysisResult>> {
    let mut all_results = Vec::new();

    // Get all valid VectorIds from the store.
    let all_ids: Vec<VectorId> = store
        .storage
        .get_sorted_serial_ids()
        .into_iter()
        .map(VectorId::from_serial_id)
        .collect();
    if all_ids.is_empty() {
        bail!("No iris codes found in store to sample from.");
    }

    // 1. Sample target codes
    let sampled_ids: Vec<VectorId> = all_ids
        .choose_multiple(rng, config.sample_size)
        .cloned()
        .collect();

    let graph = Arc::new(graph);
    // Convert to shared store so we can parallelize searches
    let store = SharedPlaintextStore::from(store);

    let mut analysis_searcher = searcher.clone();
    analysis_searcher.params.ef_search[0] = config.ef_search; // Set layer 0 `ef_search`

    let mut search_count = 0;
    let k_neighbors = config.k_neighbors;

    for target_id in sampled_ids {
        let target_code = store
            .storage
            .get_vector(&target_id)
            .await
            .ok_or_else(|| eyre!("Sampled ID {} not found in store", target_id))?
            .clone();

        let mutation_family = IrisMutationFamily::new(&target_code, rng);

        for &mutation in &config.mutations {
            let mutated_code = Arc::new(mutation_family.get_graded_similar_iris(mutation));

            let rotations = mutated_code.rotations_from_range(config.rotations.clone());
            let mut futures = Vec::new();
            for (ri, query_code_inner) in izip!(config.rotations.clone(), rotations) {
                let query_ref = Arc::new(query_code_inner);
                let analysis_searcher = analysis_searcher.clone();
                let mut store = store.clone();
                let graph = Arc::clone(&graph);

                let future = async move {
                    let neighbors = analysis_searcher
                        .search(&mut store, &graph, &query_ref, k_neighbors)
                        .await?;

                    let found = neighbors.edges.iter().any(|(id, _dist)| *id == target_id);

                    Ok(AnalysisResult {
                        id: target_id.serial_id(),
                        mutation,
                        rotation: ri,
                        found,
                    })
                };
                futures.push(future);
                search_count += 1;
            }

            // TODO: parallelize over all samples * rotations * mutations search queries,
            // instead of just rotations of some mutation
            let results_for_mutation = futures
                .into_iter()
                .map(tokio::spawn)
                .collect::<JoinAll<_>>()
                .await
                .into_iter()
                .collect::<Result<Result<Vec<_>>, JoinError>>()?;

            all_results.extend(results_for_mutation?);
            search_count = all_results.len();
        }
        if search_count % 1000 == 0 && search_count > 0 {
            println!("... performed {} searches", search_count);
        }
    }

    Ok(all_results)
}

/// Aggregates results and writes them to a CSV.
pub fn process_results(config: &AnalysisConfig, results: Vec<AnalysisResult>) -> Result<()> {
    let mut wtr = csv::Writer::from_path(&config.output_path)?;
    const PRECISION: f64 = 1000.0;

    match config.output_format.as_str() {
        "full_csv" => {
            // Option 3: Full output of individual results
            wtr.write_record(&["id", "mutation", "rotation", "found"])?;
            for res in results {
                wtr.serialize(res)?;
            }
        }
        "rate" => {
            // Option 1: Success rate for each (rotation, mutation) pair
            wtr.write_record(&["mutation", "rotation", "success_rate"])?;
            // (mutation, rotation) -> (hits, total)
            let mut rate_map: HashMap<(usize, isize), (u32, u32)> = HashMap::new();

            for res in &results {
                let key = (((res.mutation * PRECISION).floor() as usize), res.rotation);
                let entry = rate_map.entry(key).or_default();
                if res.found {
                    entry.0 += 1;
                }
                entry.1 += 1;
            }

            let mut sorted_keys: Vec<_> = rate_map.keys().into_iter().sorted().collect();
            sorted_keys.sort();

            for key in sorted_keys {
                let (hits, total) = rate_map[key];
                let rate = hits as f64 / total as f64;
                let mutation_str = format!("{}", (key.0 as f64 / PRECISION));
                wtr.write_record(&[mutation_str, key.1.to_string(), rate.to_string()])?;
            }
        }
        "histogram" => {
            // Option 2: For each rotation amount, output a histogram of "minimum mutation amount for which match was not found in search results"

            let mut min_failure_map: HashMap<(IrisSerialId, isize), usize> = HashMap::new();

            for res in results.iter() {
                let mutation_key = (res.mutation * PRECISION).floor() as usize;
                let key = (res.id, res.rotation);

                if res.found == false {
                    // Update min mutation across failures
                    let entry = min_failure_map.entry(key).or_insert(mutation_key);
                    *entry = (*entry).min(mutation_key);
                } else {
                    // Set a default of 100% mutation, in case no specified amount fails for this (rotation, id) pair
                    min_failure_map.entry(key).or_insert(PRECISION as usize);
                }
            }

            // Key: rotation -> Key: mutation_key -> Value: count_of_ids
            let mut histogram_map: BTreeMap<isize, BTreeMap<usize, u32>> = BTreeMap::new();

            // We only care about the values (rotation, min_mutation_key) from the
            // first map. The 'id' is now just a "count"
            for ((_id, rotation), min_mutation_key) in min_failure_map {
                let rotation_hist = histogram_map.entry(rotation).or_default();
                *rotation_hist.entry(min_mutation_key).or_default() += 1;
            }

            wtr.write_record(&["rotation", "min_fail_mutation", "count"])?;

            for (rotation, rotation_hist) in histogram_map {
                for (mutation_key, count) in rotation_hist {
                    let mutation_str = format!("{}", (mutation_key as f64 / PRECISION));
                    wtr.write_record(&[rotation.to_string(), mutation_str, count.to_string()])?;
                }
            }
        }
        _ => bail!("Unknown output_format: {}", config.output_format),
    }

    wtr.flush()?;
    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub irises: IrisesInit,
    pub graph: GraphInit,
    pub analysis: AnalysisConfig,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "option")]
pub enum IrisesInit {
    /// Generate N random iris codes.
    Random { number: usize },
    /// Load iris codes from a .ndjson file.
    NdjsonFile {
        path: PathBuf,
        limit: Option<usize>,
        selection: Option<IrisSelection>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "option")]
pub enum GraphInit {
    /// Build a new graph from the loaded iris codes.
    GenerateDynamic {
        size: usize,
        hnsw_params: HnswConfig,
    },
    /// Load a pre-built graph from a binary file.
    BinFile { path: PathBuf, format: GraphFormat },
}

/// HNSW parameters for graph construction.
#[derive(Debug, Deserialize)]
pub struct HnswConfig {
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m: usize,
}

// --- Helper Functions ---

/// Loads iris codes into a `PlaintextStore` based on `IrisesInit` config.
/// Returns the store and an RNG for use in later steps.
pub async fn load_iris_store(
    config: IrisesInit,
    rng: &mut StdRng,
    distance_fn: DistanceFn,
) -> Result<PlaintextStore> {
    let irises = match config {
        IrisesInit::Random { number } => {
            println!("Generating {} random iris codes...", number);
            (0..number)
                .map(|_| IrisCode::random_rng(rng))
                .collect::<Vec<_>>()
        }
        IrisesInit::NdjsonFile {
            path,
            limit,
            selection,
        } => {
            println!("Loading irises from NDJSON file: {}", path.display());
            irises_from_ndjson_iter(&path, limit, selection.unwrap_or(IrisSelection::All))?
                .collect::<Vec<_>>()
        }
    };

    let mut store = PlaintextStore::from_irises_iter(irises.into_iter());
    // Override distance_fn;
    // This is safe, because store initialization doesn't depend on distance
    store.distance_fn = distance_fn;

    Ok(store)
}

/// Loads or builds the HNSW graph.
pub async fn load_graph(
    config: &GraphInit,
    store: &mut PlaintextStore,
    rng: &mut StdRng,
) -> Result<(GraphMem<VectorId>, HnswSearcher)> {
    match config {
        GraphInit::BinFile { path, format } => {
            println!("Loading graph from binary file: {}", path.display());
            let graph = read_graph_from_file(path, *format)?;
            let searcher = HnswSearcher::new_with_test_parameters();
            Ok((graph, searcher))
        }
        GraphInit::GenerateDynamic { size, hnsw_params } => {
            println!(
                "Building new graph (size={}, M={}, ef_constr={})...",
                size, hnsw_params.m, hnsw_params.ef_construction
            );
            if *size > store.len() {
                bail!(
                    "GraphInit size ({}) is larger than loaded iris count ({})",
                    size,
                    store.len()
                );
            }
            let searcher = HnswSearcher::new(
                hnsw_params.ef_construction,
                hnsw_params.ef_search,
                hnsw_params.m,
            );

            // This method builds a graph on the first `size` entries in the store.
            // TODO: replace with parallel version
            let graph = store.generate_graph(rng, *size, &searcher).await?;

            Ok((graph, searcher))
        }
    }
}
