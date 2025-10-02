use iris_mpc_common::iris_db::iris::{fraction_less_than, IrisCode, RotExtIrisCode};
use itertools::Itertools;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPool,
};
use serde::{Deserialize, Serialize};

use crate::hawkers::plaintext_store::fraction_ordering;
use std::cmp::Ordering;

#[derive(Serialize, Deserialize)]
pub struct KNNResult {
    pub node: usize,
    neighbors: Vec<usize>,
}

pub fn naive_knn(
    irises: &[IrisCode],
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    pool.install(|| {
        (start..end)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris = &irises[i - 1];
                let mut neighbors = irises
                    .iter()
                    .enumerate()
                    .flat_map(|(j, other_iris)| {
                        (i != j + 1)
                            .then_some((j + 1, current_iris.get_distance_fraction(other_iris)))
                    })
                    .collect::<Vec<_>>();
                neighbors
                    .select_nth_unstable_by(k - 1, |lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                neighbors.shrink_to_fit(); // just to make sure
                neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                KNNResult { node: i, neighbors }
            })
            .collect::<Vec<_>>()
    })
}

pub fn naive_knn_min_fhd1(
    irises: &[[IrisCode; 31]],
    centers: &[IrisCode],
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    pool.install(|| {
        (start..end)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris = &irises[i - 1];
                let mut neighbors = centers
                    .iter()
                    .enumerate()
                    .flat_map(|(j, other_iris)| {
                        (i != j + 1).then(|| {
                            (
                                j,
                                current_iris
                                    .iter()
                                    .map(|current_rot| {
                                        current_rot.get_distance_fraction(&other_iris)
                                    })
                                    .min()
                                    .unwrap()
                                    .clone(),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                neighbors
                    .select_nth_unstable_by(k - 1, |lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                neighbors.shrink_to_fit(); // just to make sure
                neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                KNNResult { node: i, neighbors }
            })
            .collect::<Vec<_>>()
    })
}
fn fraction_difference_abs(a: (u16, u16), b: (u16, u16)) -> (u32, u32) {
    let (num1, den1) = (a.0 as u32, a.1 as u32);
    let (num2, den2) = (b.0 as u32, b.1 as u32);

    // Compute a/b - c/d = (ad - cb) / bd
    let numerator_a = num1 * den2;
    let numerator_b = num2 * den1;
    let denominator = den1 * den2;

    if numerator_a >= numerator_b {
        (numerator_a - numerator_b, denominator)
    } else {
        (numerator_b - numerator_a, denominator)
    }
}

/// Returns true if (num1/den1) <= (num2/den2), where all arguments are u32.
fn is_u32_fraction_leq_u32(num1: u32, den1: u32, num2: u32, den2: u32) -> bool {
    (num1 as u64) * (den2 as u64) <= (num2 as u64) * (den1 as u64)
}

pub fn naive_knn_min_fhd2(
    irises: &[[IrisCode; 31]],
    centers: &[IrisCode],
    self_rots: &[[(u16, u16); 31]], // Pre-computed rotation profiles
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    // B is the batch size for candidates. A larger B reduces the overhead of
    // select_nth_unstable but uses more temporary memory. Must be >= k.
    let batch_size = k;
    assert!(batch_size >= k, "Batch size must be at least k");

    pool.install(|| {
        (start..end)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris_rots = &irises[i - 1];
                let mut pruned = 0;
                // --- State for the optimization ---
                let mut best_neighbors: Vec<(usize, (u16, u16))> = vec![(0, (1, 1)); k];
                let mut candidates: Vec<(usize, (u16, u16))> = Vec::with_capacity(batch_size);
                // The threshold is the distance of the k-th best neighbor found so far
                let mut threshold = (1, 1);

                for (j, other_iris) in centers.iter().enumerate() {
                    // --- Stage 1: Fast Pruning ---

                    // a) Calculate the cheap, un-rotated distance (d0)
                    // We assume the first rotation is the un-rotated "base" iris
                    let d0 = centers[i - 1].get_distance_fraction(other_iris);

                    // b) Check if pruning is possible. If d0 is already worse than our threshold,
                    // we can invest in calculating the lower bound to see if we can skip.
                    if !fraction_less_than(&d0, &threshold) {
                        // c) Calculate the provable lower bound
                        let lower_bound = self_rots[j]
                            .iter()
                            .skip(12)
                            .take(3)
                            .map(|d_rot| fraction_difference_abs(d0, *d_rot)) // |d0 - d_rot(y,s)|
                            .sorted_by(|&lhs, &rhs| {
                                let (num1, den1) = lhs;
                                let (num2, den2) = rhs;
                                ((num1 as u64) * (den2 as u64))
                                    .cmp(&((num2 as u64) * (den1 as u64)))
                            })
                            .next()
                            .unwrap();

                        // d) The PRUNING STEP: If the best this iris can possibly be is still
                        // worse than our current k-th neighbor, skip it entirely.

                        dbg!(format!(
                            "threshold: {:.6}, lower_bound: {:.6}",
                            threshold.0 as f64 / threshold.1 as f64,
                            lower_bound.0 as f64 / lower_bound.1 as f64
                        ));
                        if is_u32_fraction_leq_u32(
                            threshold.0.into(),
                            threshold.1.into(),
                            lower_bound.0,
                            lower_bound.1,
                        ) {
                            pruned += 1;
                            continue;
                        }
                    }

                    // --- Stage 2: Exact Calculation ---
                    // If not pruned, compute the full, expensive minimum distance
                    let min_distance = current_iris_rots
                        .iter()
                        .map(|current_rot| current_rot.get_distance_fraction(other_iris))
                        .min()
                        .unwrap();

                    // Add to the batch of candidates if it's potentially better than our threshold
                    if fraction_less_than(&min_distance, &threshold) {
                        candidates.push((j, min_distance));
                    }

                    // --- Stage 3: Batch Processing ---
                    if candidates.len() >= batch_size {
                        // Combine the current best with the new candidates
                        best_neighbors.append(&mut candidates); // Drains candidates

                        // Find the new top k from the combined list
                        best_neighbors
                            .select_nth_unstable_by(k - 1, |a, b| fraction_ordering(&a.1, &b.1));
                        best_neighbors.truncate(k);

                        // Update the threshold to the distance of the new k-th neighbor (the worst of the best)
                        threshold = best_neighbors.last().unwrap().1;
                        //dbg!(threshold);
                    }
                }

                // --- Finalization ---
                // Process any remaining candidates in the last partial batch
                if !candidates.is_empty() {
                    best_neighbors.append(&mut candidates);
                    best_neighbors
                        .select_nth_unstable_by(k - 1, |a, b| fraction_ordering(&a.1, &b.1));
                    best_neighbors.truncate(k);
                }

                // Sort the final list of k neighbors by distance
                best_neighbors.sort_unstable_by(|a, b| fraction_ordering(&a.1, &b.1));

                // Extract just the indices for the final result
                let final_neighbor_indices =
                    best_neighbors.into_iter().map(|(idx, _)| idx).collect();
                dbg!({ pruned });
                KNNResult {
                    node: i,
                    neighbors: final_neighbor_indices,
                }
            })
            .collect::<Vec<_>>()
    })
}

pub fn naive_knn_min_fhd(
    irises: &[RotExtIrisCode],
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    pool.install(|| {
        (start..end)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris = &irises[i - 1];
                let mut neighbors = irises
                    .iter()
                    .enumerate()
                    .flat_map(|(j, other_iris)| {
                        (i != j + 1).then_some((j + 1, current_iris.min_fhe(other_iris)))
                    })
                    .collect::<Vec<_>>();
                neighbors
                    .select_nth_unstable_by(k - 1, |lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                neighbors.shrink_to_fit(); // just to make sure
                neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                KNNResult { node: i, neighbors }
            })
            .collect::<Vec<_>>()
    })
}
