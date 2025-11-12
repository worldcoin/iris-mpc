use crate::{
    hawkers::plaintext_store::{PlaintextStore, SharedPlaintextStore},
    hnsw::{GraphMem, HnswSearcher},
};
use eyre::{bail, eyre, Result};
use futures::future::JoinAll;
use iris_mpc_common::{IrisSerialId, IrisVectorId as VectorId};
use itertools::izip;
use rand::{rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, ops::Range, path::PathBuf, sync::Arc};
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
    /// Range of mutation amounts as (numerator, denominator).
    pub mutations: Vec<(u16, u16)>,
}

/// Struct to hold a single search result for analysis.
#[derive(Debug, Serialize)]
pub struct AnalysisResult {
    id: IrisSerialId,
    mutation_num: u16,
    mutation_denom: u16,
    rotation: isize,
    found: bool,
}
pub async fn run_analysis(
    config: &AnalysisConfig,
    store: PlaintextStore,
    graph: GraphMem<VectorId>,
    searcher: &HnswSearcher,
    //TODO: separate analysis rng from iris and graph rng
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

        for &mutation in &config.mutations {
            let mutated_code = Arc::new(target_code.get_graded_similar_iris(rng, mutation));

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
                        mutation_num: mutation.0,
                        mutation_denom: mutation.1,
                        rotation: ri,
                        found,
                    })
                };
                futures.push(future);
                search_count += 1;
            }
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

    match config.output_format.as_str() {
        "full_csv" => {
            // Option 3: Full output of individual results
            wtr.write_record(&["id", "mutation_num", "mutation_denom", "rotation", "found"])?;
            for res in results {
                wtr.serialize(res)?;
            }
        }
        "rate" => {
            // Option 1: Success rate for each (rotation, mutation) pair
            wtr.write_record(&["mutation", "rotation", "success_rate", "hits", "total"])?;
            // (mutation, rotation) -> (hits, total)
            let mut rate_map: HashMap<((u16, u16), isize), (u32, u32)> = HashMap::new();

            for res in &results {
                let key = ((res.mutation_num, res.mutation_denom), res.rotation);
                let entry = rate_map.entry(key).or_default();
                if res.found {
                    entry.0 += 1;
                }
                entry.1 += 1;
            }

            let mut sorted_keys: Vec<_> = rate_map.keys().collect();
            // Sort by rotation, then by mutation amount (n/d)
            sorted_keys.sort_by_key(|(mutation, rot)| {
                (
                    *rot,
                    (mutation.0 as f64 / mutation.1 as f64 * 1000000.0) as u32,
                )
            });

            for key in sorted_keys {
                let (hits, total) = rate_map[key];
                let rate = hits as f64 / total as f64;
                let mutation_str = format!("{}/{}", key.0 .0, key.0 .1);
                wtr.write_record(&[
                    mutation_str,
                    key.1.to_string(),
                    rate.to_string(),
                    hits.to_string(),
                    total.to_string(),
                ])?;
            }
        }
        "histogram" => {
            // Option 2: Histogram of minimum mutation amount where a match was NOT found
            wtr.write_record(&["rotation", "min_fail_mutation", "count"])?;

            // (id, rotation) -> Vec<(mutation_float, found, mutation_tuple)>
            let mut grouped_results: HashMap<(IrisSerialId, isize), Vec<(f64, bool, (u16, u16))>> =
                HashMap::new();
            for res in &results {
                let mutation_tuple = (res.mutation_num, res.mutation_denom);
                let mutation_float = res.mutation_num as f64 / res.mutation_denom as f64;
                grouped_results
                    .entry((res.id, res.rotation))
                    .or_default()
                    .push((mutation_float, res.found, mutation_tuple));
            }

            // (rotation) -> (mutation_level_tuple) -> count
            let mut hist_map: HashMap<isize, HashMap<(u16, u16), u32>> = HashMap::new();
            // Special bucket for (id, rotation) pairs that never failed
            let never_failed_key = (0, 0);

            for ((_id, rotation), mut res_list) in grouped_results {
                // Sort by mutation amount (float)
                res_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Find the first mutation that failed
                let min_fail_mutation = res_list.iter().find(|(_f, found, _t)| !*found);

                if let Some((_f, _found, mutation_tuple)) = min_fail_mutation {
                    // Increment the count for this rotation and mutation level
                    *hist_map
                        .entry(rotation)
                        .or_default()
                        .entry(*mutation_tuple)
                        .or_default() += 1;
                } else {
                    // This (id, rotation) pair *never* failed.
                    *hist_map
                        .entry(rotation)
                        .or_default()
                        .entry(never_failed_key)
                        .or_default() += 1;
                }
            }

            let mut sorted_rotations: Vec<_> = hist_map.keys().cloned().collect();
            sorted_rotations.sort();

            for rotation in sorted_rotations {
                let mut sorted_mutations: Vec<_> = hist_map[&rotation].keys().cloned().collect();
                sorted_mutations.sort_by(|a, b| {
                    let mut_a = a.0 as f64 / a.1 as f64;
                    let mut_b = b.0 as f64 / b.1 as f64;
                    mut_a
                        .partial_cmp(&mut_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for mutation in sorted_mutations {
                    let count = hist_map[&rotation][&mutation];
                    let mut_str = if mutation == never_failed_key {
                        "Never_Failed".to_string()
                    } else {
                        // Print as decimal (e.g., 0.125)
                        format!("{:.2}", mutation.0 as f64 / mutation.1 as f64)
                    };
                    wtr.write_record(&[rotation.to_string(), mut_str, count.to_string()])?;
                }
            }
        }
        _ => bail!("Unknown output_format: {}", config.output_format),
    }

    wtr.flush()?;
    Ok(())
}
