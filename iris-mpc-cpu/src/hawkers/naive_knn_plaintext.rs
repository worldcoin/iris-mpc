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

pub fn naive_knn_min_fhd2(
    irises: &[[IrisCode; 31]],
    centers: &[IrisCode],
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    let a_size = 2;
    let b_size = 64;

    let n = centers.len();
    let len = end - start;

    pool.install(|| {
        (0..(len + a_size - 1) / a_size)
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|a| {
                let left = start + a * a_size;
                let right = (start + (a + 1) * a_size).min(end);
                for b in 0..(n + b_size - 1) / b_size {
                    for i in left..right {
                        let current_iris = &irises[i];
                        let dists = (b * b_size..((b + 1) * b_size).min(n))
                            .map(|j| {
                                (
                                    j,
                                    current_iris
                                        .iter()
                                        .map(|current_rot| {
                                            current_rot.get_distance_fraction(&centers[j])
                                        })
                                        .min()
                                        .unwrap(),
                                )
                            })
                            .collect::<Vec<_>>();
                    }
                }

            //     neighbors
            //         .select_nth_unstable_by(k - 1, |lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
            //     let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
            //     neighbors.shrink_to_fit(); // just to make sure
            //     neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
            //     let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
            //     KNNResult { node: i, neighbors }
            // })
            // .collect::<Vec<_>>()
    })
}
