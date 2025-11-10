//! main.rs
//!
//! Rust binary for accuracy analysis of HNSW graph search on iris codes.
//!
//! This binary implements the following flow:
//! 1.  Read configuration from a TOML file.
//! 2.  Initialize a set of IrisCodes (either randomly or from an NDJSON file).
//! 3.  Initialize an HNSW graph (either from a pre-built file or by building from source).
//! 4.  Wraps the IrisCodes in a `VectorStore` implementation for the `HnswSearcher`.
//! 5.  Runs the accuracy analysis:
//!     a.  Samples existing codes from the store.
//!     b.  For each code, creates modified versions (mutated + rotated).
//!     c.  Searches the HNSW graph with the modified code.
//!     d.  Records if the original (target) code was found in the results.
//! 6.  Aggregates and prints the results in one of several specified formats (CSV).

use eyre::{bail, eyre, Result};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio;

// --- Assumed Crate Modules ---
// These `use` statements assume the provided files are part of a library
// crate named `hnsw_lib`.

// From iris.rs
use hnsw_lib::iris::{IrisCode, IrisCodeArray, IrisCodeBase64};

// From vector_store.rs
use hnsw_lib::vector_store::{Ref, TransientRef, VectorStore, VectorStoreMut};

// From layered_graph.rs
use hnsw_lib::layered_graph::GraphMem;

// From searcher.rs
use hnsw_lib::searcher::{HnswParams, HnswSearcher};

// --- CLI and Configuration ---

// Using `clap` for command-line argument parsing
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to the configuration TOML file
    #[arg(short, long)]
    config: PathBuf,
}

/// Top-level configuration struct, loaded from TOML.
#[derive(Deserialize, Debug)]
struct Config {
    iris_setup: IrisSetup,
    graph_setup: GraphSetup,
    analysis: Analysis,
}

#[derive(Deserialize, Debug)]
struct IrisSetup {
    /// Source of iris codes: "random" or "ndjson"
    source: String,
    /// Path to NDJSON file (if source="ndjson")
    path: Option<PathBuf>,
    /// Number of codes to generate (if source="random")
    count: Option<usize>,
}

#[derive(Deserialize, Debug)]
struct GraphSetup {
    /// Source of the graph: "binary" (load serialized) or "new" (build from iris codes)
    source: String,
    /// Path to graph file (if source="binary")
    path: Option<PathBuf>,
    /// Path to save a newly built graph (if source="new" and this is set)
    save_path: Option<PathBuf>,
    /// HNSW parameters (if source="new")
    hnsw_params: Option<HnswConfig>,
}

#[derive(Deserialize, Debug)]
struct HnswConfig {
    ef_construction: usize,
    ef_search: usize,
    m: usize,
}

#[derive(Deserialize, Debug)]
struct Analysis {
    /// Number of existing iris codes to sample for the test
    sample_size: usize,
    /// Optional seed for reproducible random sampling
    seed: Option<u64>,
    /// ef_search parameter to use during analysis
    ef_search: usize,
    /// Distance function: "fhd" (Fractional Hamming Distance) or "min_fhd" (Min-FHD)
    distance_fn: String,
    /// Number of neighbors to retrieve in search (k)
    k_neighbors: usize,
    /// Output format: "rate", "histogram", or "full_csv"
    output_format: String,
    /// Path for the output CSV file
    output_path: PathBuf,
    /// Range of relative rotations to test (e.g., [-3, -2, -1, 0, 1, 2, 3])
    rotations: Vec<isize>,
    /// Range of mutation amounts as (numerator, denominator)
    /// e.g., [[1, 32], [1, 16], [1, 8]]
    mutations: Vec<(u16, u16)>,
}

// --- VectorStore Implementation ---

/// A newtype wrapper for `usize` to be used as our `VectorRef`.
/// Implements `Ref`, `Display`, and `FromStr` as required by `GraphMem`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default, PartialOrd, Ord,
)]
struct VectorId(usize);

impl Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for VectorId {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse().map(VectorId)
    }
}

// Ensure VectorId satisfies the `Ref` trait bounds
impl Ref for VectorId {}

/// The distance function to use during search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistanceFn {
    Fhd,
    MinFhd,
}

/// An implementation of `VectorStore` that holds `IrisCode`s in memory.
#[derive(Debug)]
struct IrisVectorStore {
    /// In-memory storage of all iris codes, addressable by `VectorId`.
    codes: HashMap<VectorId, Arc<IrisCode>>,
    /// The distance function this store instance will use.
    distance_fn: DistanceFn,
}

impl IrisVectorStore {
    fn new(codes: HashMap<VectorId, Arc<IrisCode>>, distance_fn: DistanceFn) -> Self {
        Self { codes, distance_fn }
    }

    /// Helper to get a code by ID.
    fn get_code(&self, id: &VectorId) -> Result<Arc<IrisCode>> {
        self.codes
            .get(id)
            .cloned()
            .ok_or_else(|| eyre!("VectorId not found: {}", id))
    }
}

impl VectorStore for IrisVectorStore {
    /// A query is just an `Arc<IrisCode>` (the modified code).
    type QueryRef = Arc<IrisCode>;
    /// A vector reference is our `VectorId`.
    type VectorRef = VectorId;
    /// A distance is a `(u16, u16)` fraction (numerator, denominator).
    type DistanceRef = (u16, u16);

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        let vector_code = self.get_code(vector)?;

        // Calculate distance based on the configured function
        match self.distance_fn {
            DistanceFn::Fhd => Ok(query.get_distance_fraction(&vector_code)),
            DistanceFn::MinFhd => Ok(query.get_min_distance_fraction_rotation_aware(&vector_code)),
        }
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        // A match is a distance of 0, with a non-zero denominator.
        Ok(distance.0 == 0 && distance.1 > 0)
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        let (n1, d1) = *distance1;
        let (n2, d2) = *distance2;

        // Handle cases where mask overlap (denominator) is 0.
        // We treat 0/0 or n/0 as "infinite" distance.
        if d1 == 0 && d2 == 0 {
            return Ok(false); // Both infinite, not less_than
        }
        if d1 == 0 {
            return Ok(false); // d1 is infinite, d2 is finite/infinite
        }
        if d2 == 0 {
            return Ok(true); // d1 is finite, d2 is infinite
        }

        // Standard fractional comparison: n1/d1 < n2/d2  =>  n1*d2 < n2*d1
        Ok((n1 as u32) * (d2 as u32) < (n2 as u32) * (d1 as u32))
    }

    // --- Batch variants (default implementations are fine for this analysis binary) ---

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        vectors
            .into_iter()
            .filter_map(|id| self.codes.get(&id).cloned())
            .collect()
    }
}

/// We also need `VectorStoreMut` to build the graph.
impl VectorStoreMut for IrisVectorStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // This is a bit of a hack for graph building.
        // We assume the query code is already in `self.codes` and find its ID.
        // A real insertion would add a new entry.
        for (id, code) in &self.codes {
            if Arc::ptr_eq(code, query) {
                return *id;
            }
        }
        // If not found (which shouldn't happen in our "new" graph flow),
        // we'd have to add it.
        let new_id = VectorId(self.codes.len());
        self.codes.insert(new_id, query.clone());
        new_id
    }

    async fn insert_at(
        &mut self,
        _vector_ref: &Self::VectorRef,
        _query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        unimplemented!("insert_at is not needed for this analysis")
    }
}

// --- Main Execution ---

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse CLI and Config
    let cli = Cli::parse();
    let config_toml = std::fs::read_to_string(&cli.config)?;
    let config: Config = toml::from_str(&config_toml)?;

    println!("Configuration loaded.");

    // 2. Initialize Iris Codes
    let (iris_codes, mut rng) = load_iris_codes(&config.iris_setup, config.analysis.seed)?;
    println!("Loaded {} iris codes.", iris_codes.len());

    // 3. Initialize VectorStore
    let distance_fn = match config.analysis.distance_fn.as_str() {
        "fhd" => DistanceFn::Fhd,
        "min_fhd" => DistanceFn::MinFhd,
        _ => bail!("Unknown distance_fn: {}", config.analysis.distance_fn),
    };
    let mut store = IrisVectorStore::new(iris_codes, distance_fn);

    // 4. Initialize Graph
    let (graph, searcher) = load_graph(&config.graph_setup, &mut store, &mut rng).await?;
    println!("Graph initialized.");

    // 5. Execute Analysis
    println!("Starting analysis...");
    let results = run_analysis(&config.analysis, &mut store, &graph, &searcher, &mut rng).await?;
    println!("Analysis complete. {} searches performed.", results.len());

    // 6. Print Aggregated Output
    process_results(&config.analysis, results)?;
    println!(
        "Results written to {}.",
        config.analysis.output_path.display()
    );

    Ok(())
}

// --- Helper Functions ---

/// Loads iris codes based on `IrisSetup` config.
fn load_iris_codes(
    config: &IrisSetup,
    seed: Option<u64>,
) -> Result<(HashMap<VectorId, Arc<IrisCode>>, StdRng)> {
    let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
    let mut codes = HashMap::new();

    match config.source.as_str() {
        "random" => {
            let count = config
                .count
                .ok_or_eyre("`count` must be set for random iris source")?;
            for i in 0..count {
                codes.insert(VectorId(i), Arc::new(IrisCode::random_rng(&mut rng)));
            }
        }
        "ndjson" => {
            let path = config
                .path
                .as_ref()
                .ok_or_eyre("`path` must be set for ndjson iris source")?;
            let file = File::open(path)?;
            for (i, line) in BufReader::new(file).lines().enumerate() {
                let base64_code: IrisCodeBase64 = serde_json::from_str(&line?)?;
                codes.insert(VectorId(i), Arc::new(IrisCode::from(&base64_code)));
            }
        }
        _ => bail!("Unknown iris_setup.source: {}", config.source),
    }

    Ok((codes, rng))
}

/// Loads or builds the HNSW graph.
async fn load_graph(
    config: &GraphSetup,
    store: &mut IrisVectorStore,
    rng: &mut StdRng,
) -> Result<(GraphMem<VectorId>, HnswSearcher)> {
    match config.source.as_str() {
        "binary" => {
            let path = config
                .path
                .as_ref()
                .ok_or_eyre("`path` must be set for binary graph source")?;
            println!("Loading graph from binary: {}", path.display());
            let file = File::open(path)?;
            let graph: GraphMem<VectorId> = bincode::deserialize_from(BufReader::new(file))?;

            // Assume default params for searcher, as they aren't stored with graph
            let searcher = HnswSearcher::new_with_test_parameters();
            Ok((graph, searcher))
        }
        "new" => {
            let params = config
                .hnsw_params
                .as_ref()
                .ok_or_eyre("`hnsw_params` must be set for new graph source")?;
            println!(
                "Building new graph (M={}, ef_constr={})...",
                params.m, params.ef_construction
            );

            let searcher = HnswSearcher::new(params.ef_construction, params.ef_search, params.m);
            let mut graph = GraphMem::new();

            let all_codes: Vec<(VectorId, Arc<IrisCode>)> = store
                .codes
                .iter()
                .map(|(id, code)| (*id, code.clone()))
                .collect();

            for (id, code) in all_codes {
                let insertion_layer = searcher.select_layer_rng(rng)?;
                let (neighbors, set_ep) = searcher
                    .search_to_insert(store, &graph, &code, insertion_layer)
                    .await?;
                // Note: `store.insert` is a hack that just finds the ID.
                let inserted_ref = store.insert(&code).await;
                assert_eq!(id, inserted_ref); // Sanity check

                searcher
                    .insert_from_search_results(store, &mut graph, inserted_ref, neighbors, set_ep)
                    .await?;
            }

            if let Some(save_path) = &config.save_path {
                println!("Saving newly built graph to {}", save_path.display());
                let file = File::create(save_path)?;
                bincode::serialize_into(file, &graph)?;
            }

            Ok((graph, searcher))
        }
        _ => bail!("Unknown graph_setup.source: {}", config.source),
    }
}

/// Struct to hold a single search result for analysis.
#[derive(Debug, Serialize)]
struct AnalysisResult {
    id: VectorId,
    mutation_num: u16,
    mutation_denom: u16,
    rotation: isize,
    found: bool,
}

/// Runs the main analysis loop.
async fn run_analysis(
    config: &Analysis,
    store: &mut IrisVectorStore,
    graph: &GraphMem<VectorId>,
    searcher: &HnswSearcher,
    rng: &mut StdRng,
) -> Result<Vec<AnalysisResult>> {
    let mut all_results = Vec::new();
    let all_ids: Vec<VectorId> = store.codes.keys().cloned().collect();

    // 1. Sample target codes
    let sampled_ids: Vec<VectorId> = all_ids
        .choose_multiple(rng, config.sample_size)
        .cloned()
        .collect();

    // Configure the searcher's `ef_search` parameter for the analysis
    let mut analysis_searcher = searcher.clone();
    // This is a bit awkward as HnswParams is complex. We'll just set layer 0.
    analysis_searcher.params.ef_search[0] = config.ef_search;

    let mut search_count = 0;

    // 2. Outer loop: Iterate through sampled target codes
    for target_id in sampled_ids {
        let target_code = store.get_code(&target_id)?;

        // 3. Middle loop: Iterate through mutation levels
        for &mutation in &config.mutations {
            let mutated_code = target_code.get_graded_similar_iris(rng, mutation);

            // 4. Inner loop: Iterate through rotations
            for &rotation in &config.rotations {
                let mut query_code = mutated_code.clone();
                if rotation > 0 {
                    query_code.rotate_right(rotation as usize);
                } else {
                    query_code.rotate_left(-rotation as usize);
                }

                // 5. Perform the search
                let neighbors = analysis_searcher
                    .search(store, graph, &Arc::new(query_code), config.k_neighbors)
                    .await?;

                // 6. Check if the target was found
                let found = neighbors.edges.iter().any(|(id, _dist)| *id == target_id);

                all_results.push(AnalysisResult {
                    id: target_id,
                    mutation_num: mutation.0,
                    mutation_denom: mutation.1,
                    rotation,
                    found,
                });
                search_count += 1;
            }
        }
        if search_count % 1000 == 0 && search_count > 0 {
            println!("... performed {} searches", search_count);
        }
    }

    Ok(all_results)
}

// /// Aggregates results and writes them to a CSV.
// fn process_results(config: &Analysis, results: Vec<AnalysisResult>) -> Result<()> {
//     let mut wtr = csv::Writer::from_path(&config.output_path)?;

//     match config.output_format.as_str() {
//         "full_csv" => {
//             wtr.write_record(&[
//                 "id",
//                 "mutation_num",
//                 "mutation_denom",
//                 "rotation",
//                 "found",
//             ])?;
//             for res in results {
//                 wtr.serialize(res)?;
//             }
//         }
//         "rate" => {
//             wtr.write_record(&["mutation", "rotation", "success_rate", "hits", "total"])?;
//             // (mutation, rotation) -> (hits, total)
//             let mut rate_map: HashMap<((u16, u16), isize), (u32, u32)> = HashMap::new();

//             for res in results {
//                 let key = ((res.mutation_num, res.mutation_denom), res.rotation);
//                 let entry = rate_map.entry(key).or_default();
//                 if res.found {
//                     entry.0 += 1;
//                 }
//                 entry.1 += 1;
//             }

//             let mut sorted_keys: Vec<_> = rate_map.keys().collect();
//             sorted_keys.sort_by_key(|(mut, rot)| {
//                 (
//                     *rot,
//                     (mut.0 as f64 / mut.1 as f64 * 1000000.0) as u32,
//                 )
//             });

//             for key in sorted_keys {
//                 let (hits, total) = rate_map[key];
//                 let rate = hits as f64 / total as f64;
//                 let mutation_str = format!("{}/{}", key.0.0, key.0.1);
//                 wtr.write_record(&[
//                     mutation_str,
//                     key.1.to_string(),
//                     rate.to_string(),
//                     hits.to_string(),
//                     total.to_string(),
//                 ])?;
//             }
//         }
//         "histogram" => {
//             wtr.write_record(&["rotation", "min_fail_mutation", "count"])?;
//             // Find the minimum mutation amount *per (id, rotation)* where a match was NOT found.
//             // (id, rotation) -> Vec<(mutation_float, found)>
//             let mut grouped_results: HashMap<(VectorId, isize), Vec<((u16, u16), bool)>> = HashMap::new();
//             for res in results {
//                 let mutation = (res.mutation_num, res.mutation_denom);
//                 grouped_results.entry((res.id, res.rotation)).or_default().push((mutation, res.found));
//             }

//             // (rotation) -> (mutation_level) -> count
//             let mut hist_map: HashMap<isize, HashMap<(u16, u16), u32>> = HashMap::new();

//             for ((_id, rotation), mut res_list) in grouped_results {
//                 // Sort by mutation amount (n/d)
//                 res_list.sort_by(|a, b| {
//                     let mut_a = a.0.0 as f64 / a.0.1 as f64;
//                     let mut_b = b.0.0 as f64 / b.0.1 as f64;
//                     mut_a.partial_cmp(&mut_b).unwrap_or(std::cmp::Ordering::Equal)
//                 });

//                 // Find the first mutation that failed
//                 let min_fail_mutation = res_list.iter().find(|(_mut, found)| !*found);

//                 if let Some((mutation, _found)) = min_fail_mutation {
//                      *hist_map.entry(rotation).or_default().entry(*mutation).or_default() += 1;
//                 } else {
//                     // This (id, rotation) pair *never* failed.
//                     // We can log this as a special "Never_Failed" bucket.
//                     *hist_map.entry(rotation).or_default().entry((0, 1)).or_default() += 1; // (0, 1) means "Never_Failed"
//                 }
//             }

//             let mut sorted_rotations: Vec<_> = hist_map.keys().cloned().collect();
//             sorted_rotations.sort();

//             for rotation in sorted_rotations {
//                 let mut sorted_mutations: Vec<_> = hist_map[&rotation].keys().cloned().collect();
//                 sorted_mutations.sort_by(|a, b| {
//                      let mut_a = a.0 as f64 / a.1 as f64;
//                      let mut_b = b.0 as f64 / b.1 as f64;
//                      mut_a.partial_cmp(&mut_b).unwrap_or(std.cmp::Ordering::Equal)
//                 });

//                 for mutation in sorted_mutations {
//                     let count = hist_map[&rotation][&mutation];
//                     let mut_str = if mutation == (0, 1) {
//                         "Never_Failed".to_string()
//                     } else {
//                         format!("{}/{}", mutation.0, mutation.1)
//                     };
//                     wtr.write_record(&[
//                         rotation.to_string(),
//                         mut_str,
//                         count.to_string()
//                     ])?;
//                 }
//             }
//         }
//         _ => bail!("Unknown output_format: {}", config.output_format),
//     }

//     wtr.flush()?;
//     Ok(())
// }
