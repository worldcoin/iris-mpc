use std::cmp::Ordering;

use clap::ValueEnum;
use iris_mpc_common::iris_db::iris::IrisCode;
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};
use serde::{Deserialize, Serialize};

use crate::hawkers::plaintext_store::fraction_ordering;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct KNNResult {
    pub node: usize,
    neighbors: Vec<usize>,
}

pub trait KNNEngine {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: usize, num_threads: usize) -> Self;
    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult>;
}

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineChoice {
    NaiveFHD,
    NaiveMinFHD,
}
pub enum Engine {
    NaiveFHD(NaiveNormalDistKNN),
    NaiveMinFHD(NaiveMinFHDKNN),
}

impl Engine {
    pub fn init(
        which: EngineChoice,
        irises: Vec<IrisCode>,
        k: usize,
        next_id: usize,
        num_threads: usize,
    ) -> Self {
        match which {
            EngineChoice::NaiveFHD => {
                Self::NaiveFHD(NaiveNormalDistKNN::init(irises, k, next_id, num_threads))
            }
            EngineChoice::NaiveMinFHD => {
                Self::NaiveMinFHD(NaiveMinFHDKNN::init(irises, k, next_id, num_threads))
            }
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        match self {
            Self::NaiveFHD(engine) => engine.compute_chunk(chunk_size),
            Self::NaiveMinFHD(engine) => engine.compute_chunk(chunk_size),
        }
    }

    /// Next id to process
    pub fn next_id(&self) -> usize {
        match self {
            Self::NaiveFHD(engine) => engine.next_id,
            Self::NaiveMinFHD(engine) => engine.next_id,
        }
    }
}

pub struct NaiveNormalDistKNN {
    irises: Vec<IrisCode>,
    k: usize,
    next_id: usize,
    pool: ThreadPool,
}

impl KNNEngine for NaiveNormalDistKNN {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: usize, num_threads: usize) -> Self {
        NaiveNormalDistKNN {
            irises,
            k,
            next_id,
            pool: ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap(),
        }
    }

    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        let start = self.next_id;
        let end = (start + chunk_size).min(self.irises.len() + 1);
        self.next_id = end;

        self.pool.install(|| {
            (start..end)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|i| {
                    let current_iris = &self.irises[i - 1];
                    let mut neighbors = self
                        .irises
                        .iter()
                        .enumerate()
                        .flat_map(|(j, other_iris)| {
                            (i != j + 1)
                                .then_some((j + 1, current_iris.get_distance_fraction(other_iris)))
                        })
                        .collect::<Vec<_>>();
                    neighbors.select_nth_unstable_by(self.k - 1, |lhs, rhs| {
                        fraction_ordering(&lhs.1, &rhs.1)
                    });
                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit(); // just to make sure
                    neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                    let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                    KNNResult { node: i, neighbors }
                })
                .collect::<Vec<_>>()
        })
    }
}

pub struct NaiveMinFHDKNN {
    irises_with_rotations: Vec<[IrisCode; 31]>,
    centers: Vec<IrisCode>, // Store centers separately to access them contiguously
    k: usize,
    next_id: usize,
    pool: ThreadPool,
}

impl KNNEngine for NaiveMinFHDKNN {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: usize, num_threads: usize) -> Self {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        let irises_with_rotations = pool.install(|| {
            irises
                .par_iter()
                .map(|iris| iris.all_rotations().try_into().unwrap())
                .collect::<Vec<_>>()
        });
        NaiveMinFHDKNN {
            irises_with_rotations,
            centers: irises,
            k,
            next_id,
            pool,
        }
    }

    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        let start = self.next_id;
        let end = (start + chunk_size).min(self.centers.len() + 1);
        self.next_id = end;

        self.pool.install(|| {
            (start..end)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|i| {
                    let current_iris = &self.irises_with_rotations[i - 1];
                    let mut neighbors = self
                        .centers
                        .iter()
                        .enumerate()
                        .flat_map(|(j, other_iris)| {
                            (i != j + 1).then(|| {
                                (
                                    j + 1,
                                    current_iris
                                        .iter()
                                        .map(|current_rot| {
                                            current_rot.get_distance_fraction(other_iris)
                                        })
                                        .min()
                                        .unwrap(),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    neighbors.select_nth_unstable_by(
                        self.k - 1,
                        |lhs, rhs| match fraction_ordering(&lhs.1, &rhs.1) {
                            Ordering::Equal => lhs.0.cmp(&rhs.0),
                            other => other,
                        },
                    );
                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit(); // just to make sure
                    neighbors.sort_by(|lhs, rhs| match fraction_ordering(&lhs.1, &rhs.1) {
                        Ordering::Equal => lhs.0.cmp(&rhs.0),
                        other => other,
                    });
                    let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                    KNNResult { node: i, neighbors }
                })
                .collect::<Vec<_>>()
        })
    }
}

// pub fn naive_knn_min_fhd2(
//     irises: &[[IrisCode; 31]],
//     centers: &[IrisCode],
//     k: usize,
//     start: usize,
//     end: usize,
//     pool: &ThreadPool,
// ) -> Vec<KNNResult> {
//     let n = centers.len();
//     let len = end - start;

//     // Chosen for L2: 5MiB, L1d: 192 KiB
//     // One iris is 3.2 KB
//     // One iris with all rotations is 99.2 KB
//     // 25 * 99.2 KB ~= 2.4 MB <= 5 Mib
//     // 32 * 3.2 KB ~= 102 KB <= 192 Kib
//     let a_size = 25;
//     let b_size = 48;

//     pool.install(|| {
//         (0..=(len - 1) / a_size)
//             .collect::<Vec<_>>()
//             .into_par_iter()
//             .flat_map(|a| {
//                 let left = start + a * a_size;
//                 let right = (start + (a + 1) * a_size).min(end);

//                 let capacity = 2 * k + b_size;
//                 let mut results = vec![Vec::with_capacity(capacity); right - left];

//                 for b in 0..=(n - 1) / b_size {
//                     for i in left..right {
//                         #[allow(clippy::needless_range_loop)]
//                         for j in b * b_size..((b + 1) * b_size).min(n) {
//                             if i != j + 1 {
//                                 let dist = irises[i - 1]
//                                     .iter()
//                                     .map(|current_rot| {
//                                         current_rot.get_distance_fraction(&centers[j])
//                                     })
//                                     .min()
//                                     .unwrap();
//                                 results[i - left].push((j + 1, dist));
//                             }
//                         }
//                     }

//                     // Generally all lengths are equal, except for the cases generated by i != j
//                     // Judging it better to branch only once for everyone, with the cost that it can
//                     // rarely delay the truncation
//                     if results[0].len() >= 2 * k {
//                         for i in left..right {
//                             results[i - left].select_nth_unstable_by(k - 1, |lhs, rhs| {
//                                 match fraction_ordering(&lhs.1, &rhs.1) {
//                                     Ordering::Equal => lhs.0.cmp(&rhs.0),
//                                     other => other,
//                                 }
//                             });
//                             results[i - left].truncate(k);
//                         }
//                     }
//                 }

//                 results
//                     .into_iter()
//                     .enumerate()
//                     .map(|(i, mut result)| {
//                         if result.len() > k {
//                             result.select_nth_unstable_by(
//                                 k - 1,
//                                 |lhs, rhs| match fraction_ordering(&lhs.1, &rhs.1) {
//                                     Ordering::Equal => lhs.0.cmp(&rhs.0),
//                                     other => other,
//                                 },
//                             );
//                             result.truncate(k);
//                         }
//                         result.shrink_to_fit();
//                         result.sort_by(|lhs, rhs| match fraction_ordering(&lhs.1, &rhs.1) {
//                             Ordering::Equal => lhs.0.cmp(&rhs.0),
//                             other => other,
//                         });

//                         KNNResult {
//                             node: left + i,
//                             neighbors: result.into_iter().map(|(j, _)| j).collect::<Vec<_>>(),
//                         }
//                     })
//                     .collect::<Vec<_>>()
//             })
//             .collect::<Vec<_>>()
//     })
// }
