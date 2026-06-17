//! Accuracy analysis for `PlaintextDeepIDStore`, parallel to `analysis::accuracy`.
//!
//! No rotations / mutations: deep-ID vectors are perturbed by adding ±1 to
//! each nibble independently with probability `noise_level`, clamping to the
//! supported `{-8..=7}` domain.

use crate::{
    hawkers::plaintext_deep_id_store::{
        Int4Vector, PlaintextDeepIDStore, SharedPlaintextDeepIDStore, INT4_DIM,
    },
    hnsw::{
        searcher::{LayerDistribution, LayerMode, N_PARAM_LAYERS},
        GraphMem, HnswParams, HnswSearcher, SortedNeighborhood,
    },
    utils::serialization::{
        graph::{read_graph_from_file, GraphFormat},
        int4_ndjson::int4_vectors_from_ndjson_iter,
    },
};
use eyre::{bail, eyre, Result};
use iris_mpc_common::{IrisSerialId, IrisVectorId as VectorId};
use itertools::Itertools;
use rand::{rngs::StdRng, seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    future::Future,
    path::PathBuf,
    sync::Arc,
};
use tracing::{info_span, Instrument};

/* ----------------------------- Config ----------------------------- */

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub vectors: Int4VectorsInit,
    pub graph: GraphInit,
    pub analysis: AnalysisConfig,
    pub threshold: i32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "option")]
pub enum Int4VectorsInit {
    Random { number: usize },
    NdjsonFile { path: PathBuf, limit: Option<usize> },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "option")]
pub enum GraphInit {
    GenerateDynamic {
        size: usize,
        gen_hnsw_config: HnswConfig,
    },
    BinFile {
        path: PathBuf,
        format: GraphFormat,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnalysisConfig {
    pub sample_size: usize,
    pub seed: Option<u64>,
    pub k_neighbors: usize,
    pub output_format: String, // "rate" | "full_csv" | "histogram"
    pub output_path: PathBuf,
    pub metrics_path: Option<PathBuf>,
    pub noise_levels: Vec<f64>,
    pub search_hnsw_config: HnswConfig,
}

impl AnalysisConfig {
    /// Validate that every entry of `noise_levels` is a finite number in `[0, 1]`.
    /// The seed derivation and bucket-keying downstream assume this domain;
    /// out-of-range values would silently collide into the same bucket.
    pub fn validate(&self) -> Result<()> {
        for (i, &p) in self.noise_levels.iter().enumerate() {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                bail!(
                    "noise_levels[{}] = {} is not a finite value in [0, 1]",
                    i,
                    p
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum LayerValue<T> {
    Single(T),
    PerLayer(Vec<T>),
}

#[derive(Clone, Debug, Deserialize)]
#[allow(non_snake_case)]
pub struct HnswConfig {
    pub ef_construction: usize,
    pub ef_search: LayerValue<usize>,
    pub M: usize,
    pub layer_mode: LayerMode,
    #[serde(default)]
    pub fixed_layer_search_batch_size: Option<usize>,
}

impl From<&HnswConfig> for HnswSearcher {
    fn from(value: &HnswConfig) -> Self {
        let ef_search_first = *match &value.ef_search {
            LayerValue::Single(val) => val,
            LayerValue::PerLayer(vals) => vals.first().unwrap(),
        };

        let mut params = HnswParams::new(value.ef_construction, ef_search_first, value.M);

        if let LayerValue::PerLayer(vals) = &value.ef_search {
            let mut vals = vals.clone();
            if vals.len() < N_PARAM_LAYERS {
                let last = *vals.last().unwrap();
                vals.resize(N_PARAM_LAYERS, last);
            }
            params.ef_search = vals.clone().try_into().unwrap();
            params.ef_constr_search = vals.try_into().unwrap();
        }

        let layer_mode = value.layer_mode.clone();
        let layer_distribution = LayerDistribution::new_geometric_from_M(value.M);

        HnswSearcher {
            params,
            layer_mode,
            layer_distribution,
            fixed_layer_search_batch_size: value.fixed_layer_search_batch_size,
        }
    }
}

/* ----------------------------- Loading ----------------------------- */

pub async fn load_deep_id_store(
    config: Int4VectorsInit,
    threshold: i32,
    rng: &mut StdRng,
) -> Result<PlaintextDeepIDStore> {
    let vectors = match config {
        Int4VectorsInit::Random { number } => {
            println!("Generating {} random Int4Vectors...", number);
            (0..number)
                .map(|_| Int4Vector::random(rng))
                .collect::<Vec<_>>()
        }
        Int4VectorsInit::NdjsonFile { path, limit } => {
            println!("Loading Int4Vectors from NDJSON file: {}", path.display());
            int4_vectors_from_ndjson_iter(&path, limit)?.collect::<Result<Vec<_>>>()?
        }
    };

    let mut store = PlaintextDeepIDStore::new(threshold);
    for (idx, v) in vectors.into_iter().enumerate() {
        let id = VectorId::from_serial_id((idx + 1) as u32);
        store.insert_with_id(id, Arc::new(v));
    }
    Ok(store)
}

pub async fn load_graph(
    config: &GraphInit,
    store: &mut PlaintextDeepIDStore,
    rng: &mut StdRng,
) -> Result<GraphMem> {
    match config {
        GraphInit::BinFile { path, format } => {
            println!("Loading graph from binary file: {}", path.display());
            Ok(read_graph_from_file(path, *format)?)
        }
        GraphInit::GenerateDynamic {
            size,
            gen_hnsw_config,
        } => {
            if *size > store.len() {
                bail!(
                    "GraphInit size ({}) is larger than loaded vector count ({})",
                    size,
                    store.len()
                );
            }
            let searcher: HnswSearcher = gen_hnsw_config.into();
            let graph = store.generate_graph(rng, *size, &searcher).await?;
            Ok(graph)
        }
    }
}

/* ----------------------------- Perturbation ----------------------------- */

/// Perturb each nibble of `v` independently: with probability `p`, add a
/// uniformly chosen ±1, clamping to the supported `{-8..=7}` domain.
pub fn perturb_nibbles<R: Rng>(v: &Int4Vector, p: f64, rng: &mut R) -> Int4Vector {
    let mut out = v.clone();
    for i in 0..INT4_DIM {
        if rng.gen::<f64>() < p {
            let delta: i8 = if rng.gen::<bool>() { 1 } else { -1 };
            let new = (out.get(i) + delta).clamp(-8, 7);
            out.set(i, new);
        }
    }
    out
}

/* ----------------------------- Analysis ----------------------------- */

#[derive(Debug, Serialize)]
pub struct AnalysisResult {
    id: IrisSerialId,
    noise_level: f64,
    found: bool,
}

async fn execute_batch<F>(futures: Vec<F>) -> Result<Vec<AnalysisResult>>
where
    F: Future<Output = Result<AnalysisResult>> + Send + 'static,
{
    let handles = futures.into_iter().map(tokio::spawn).collect_vec();
    let join_results = futures::future::join_all(handles).await;

    let mut results = Vec::with_capacity(join_results.len());
    for join_result in join_results {
        results.push(join_result??);
    }
    Ok(results)
}

pub async fn run_analysis(
    config: AnalysisConfig,
    store: PlaintextDeepIDStore,
    graph: GraphMem,
    rng: &mut StdRng,
) -> Result<Vec<AnalysisResult>> {
    config.validate()?;

    let all_ids: Vec<VectorId> = store
        .storage
        .get_sorted_serial_ids()
        .into_iter()
        .map(VectorId::from_serial_id)
        .collect();
    if all_ids.is_empty() {
        bail!("No deep-ID vectors found in store to sample from.");
    }

    let sampled_ids: Vec<VectorId> = all_ids
        .choose_multiple(rng, config.sample_size)
        .cloned()
        .collect();

    let graph = Arc::new(graph);
    let store = SharedPlaintextDeepIDStore::from(store);

    let analysis_searcher: HnswSearcher = (&config.search_hnsw_config).into();

    let k_neighbors = config.k_neighbors;
    let mut futures = Vec::new();

    let total_queries = config.sample_size * config.noise_levels.len();
    println!("Processing {} search queries...", total_queries);

    const CHUNK_SIZE: usize = 200;
    let mut all_results = Vec::with_capacity(total_queries);

    for &target_id in &sampled_ids {
        let target_vec = store
            .storage
            .get_vector(&target_id)
            .await
            .ok_or_else(|| eyre!("Sampled ID {} not found in store", target_id))?;

        for &noise_level in &config.noise_levels {
            // f64::to_bits gives a unique u64 per distinct noise_level value,
            // avoiding collisions when two configured levels round to the same
            // integer.
            let mut local_rng: StdRng = rand::SeedableRng::seed_from_u64(
                ((target_id.serial_id() as u64) << 32) ^ noise_level.to_bits(),
            );
            let query_inner = perturb_nibbles(&target_vec, noise_level, &mut local_rng);
            let query_ref = Arc::new(query_inner);

            let analysis_searcher_clone = analysis_searcher.clone();
            let mut store_clone = store.clone();
            let graph_clone = Arc::clone(&graph);

            let future = async move {
                let neighbors: SortedNeighborhood<_> = analysis_searcher_clone
                    .search(&mut store_clone, &graph_clone, &query_ref, k_neighbors)
                    .await?;

                let found = neighbors
                    .as_ref()
                    .iter()
                    .any(|(id, _dist)| *id == target_id);

                eyre::Ok(AnalysisResult {
                    id: target_id.serial_id(),
                    noise_level,
                    found,
                })
            }
            .instrument(info_span!(
                "search_task",
                __query_id = target_id.serial_id(),
                __noise = noise_level,
            ));
            futures.push(future);

            if futures.len() >= CHUNK_SIZE {
                let batch_results = execute_batch(std::mem::take(&mut futures)).await?;
                all_results.extend(batch_results);
                println!("Processed {} queries...", all_results.len());
            }
        }
    }

    let remaining_results = execute_batch(futures).await?;
    all_results.extend(remaining_results);

    println!("... all {} searches complete.", all_results.len());
    Ok(all_results)
}

pub fn process_results(config: &AnalysisConfig, results: Vec<AnalysisResult>) -> Result<()> {
    let mut wtr = csv::Writer::from_path(&config.output_path)?;
    const PRECISION: f64 = 1000.0;

    match config.output_format.as_str() {
        "full_csv" => {
            wtr.write_record(["id", "noise_level", "found"])?;
            for res in results {
                wtr.serialize(res)?;
            }
        }
        "rate" => {
            wtr.write_record(["noise_level", "success_rate"])?;
            let mut rate_map: HashMap<usize, (u32, u32)> = HashMap::new();
            for res in &results {
                let key = (res.noise_level * PRECISION).floor() as usize;
                let entry = rate_map.entry(key).or_default();
                if res.found {
                    entry.0 += 1;
                }
                entry.1 += 1;
            }
            let sorted_keys: Vec<_> = rate_map.keys().sorted().collect();
            for key in sorted_keys {
                let (hits, total) = rate_map[key];
                let rate = hits as f64 / total as f64;
                let noise_str = format!("{}", (*key as f64 / PRECISION));
                wtr.write_record(&[noise_str, rate.to_string()])?;
            }
        }
        "histogram" => {
            let mut min_failure_map: HashMap<IrisSerialId, usize> = HashMap::new();
            for res in results.iter() {
                let key = (res.noise_level * PRECISION).floor() as usize;
                if !res.found {
                    let entry = min_failure_map.entry(res.id).or_insert(key);
                    *entry = (*entry).min(key);
                } else {
                    min_failure_map.entry(res.id).or_insert(PRECISION as usize);
                }
            }
            let mut histogram: BTreeMap<usize, u32> = BTreeMap::new();
            #[allow(
                clippy::iter_over_hash_type,
                reason = "we use the key for inserting, order does not matter"
            )]
            for (_id, min_key) in min_failure_map {
                *histogram.entry(min_key).or_default() += 1;
            }
            wtr.write_record(["min_fail_noise_level", "count"])?;
            for (key, count) in histogram {
                wtr.write_record([format!("{}", (key as f64 / PRECISION)), count.to_string()])?;
            }
        }
        _ => bail!("Unknown output_format: {}", config.output_format),
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::searcher::LayerMode;
    use aes_prng::AesRng;
    use rand::SeedableRng;

    fn small_hnsw_config() -> HnswConfig {
        HnswConfig {
            ef_construction: 32,
            ef_search: LayerValue::Single(32),
            M: 16,
            layer_mode: LayerMode::LinearScan { max_graph_layer: 1 },
            fixed_layer_search_batch_size: None,
        }
    }

    #[tokio::test]
    async fn noise_zero_finds_self() {
        let mut std_rng: StdRng = SeedableRng::seed_from_u64(0);
        let store = load_deep_id_store(
            Int4VectorsInit::Random { number: 32 },
            /* threshold */ 0,
            &mut std_rng,
        )
        .await
        .unwrap();

        let mut aes_rng = AesRng::seed_from_u64(0);
        let searcher: HnswSearcher = (&small_hnsw_config()).into();
        let mut store_for_graph = store.clone();
        let graph = store_for_graph
            .generate_graph(&mut aes_rng, store_for_graph.len(), &searcher)
            .await
            .unwrap();

        let analysis = AnalysisConfig {
            sample_size: 16,
            seed: Some(0),
            k_neighbors: 1,
            output_format: "rate".into(),
            output_path: PathBuf::from("/tmp/__deep_id_test_output.csv"),
            metrics_path: None,
            noise_levels: vec![0.0],
            search_hnsw_config: small_hnsw_config(),
        };

        let mut analysis_rng: StdRng = SeedableRng::seed_from_u64(0);
        let results = run_analysis(analysis, store_for_graph, graph, &mut analysis_rng)
            .await
            .unwrap();

        assert_eq!(results.len(), 16);
        let n_found = results.iter().filter(|r| r.found).count();
        assert_eq!(n_found, 16, "noise=0 self-search must always find target");
    }
}
