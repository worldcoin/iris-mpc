use std::{
    cmp::Ordering,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use clap::ValueEnum;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId};
use rayon::{
    iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};
use serde::{Deserialize, Serialize};

use crate::hawkers::plaintext_store::fraction_ordering;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct KNNResult<V> {
    pub node: V,
    pub neighbors: Vec<V>,
}

impl<U> KNNResult<U> {
    pub fn map<V, F>(self, mut f: F) -> KNNResult<V>
    where
        F: FnMut(U) -> V,
    {
        KNNResult {
            node: f(self.node),
            neighbors: self.neighbors.into_iter().map(f).collect(),
        }
    }
}

/// Reads a Vec<KNNResult<u32>> from a file, skipping the first line (header).
pub fn read_knn_results_from_file(path: PathBuf) -> std::io::Result<Vec<KNNResult<u32>>> {
    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    // Skip the header
    lines.next();

    let mut results = Vec::new();
    for line in lines {
        let line = line?;
        let knn_result: KNNResult<u32> = serde_json::from_str(&line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        results.push(knn_result);
    }
    Ok(results)
}

pub trait KNNEngine {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: IrisSerialId) -> Self;
    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>>;
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
        next_id: IrisSerialId,
    ) -> Self {
        assert!(k < irises.len());
        match which {
            EngineChoice::NaiveFHD => Self::NaiveFHD(NaiveNormalDistKNN::init(irises, k, next_id)),
            EngineChoice::NaiveMinFHD => {
                Self::NaiveMinFHD(NaiveMinFHDKNN::init(irises, k, next_id))
            }
        }
    }

    // TODO refactor this using the new distance_fn object (?)
    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        match self {
            Self::NaiveFHD(engine) => engine.compute_chunk(chunk_size),
            Self::NaiveMinFHD(engine) => engine.compute_chunk(chunk_size),
        }
    }

    /// Next id to process
    pub fn next_id(&self) -> IrisSerialId {
        match self {
            Self::NaiveFHD(engine) => engine.next_id,
            Self::NaiveMinFHD(engine) => engine.next_id,
        }
    }
}

pub struct NaiveNormalDistKNN {
    irises: Vec<IrisCode>,
    k: usize,
    next_id: IrisSerialId,
    pool: ThreadPool,
}

impl KNNEngine for NaiveNormalDistKNN {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: IrisSerialId) -> Self {
        NaiveNormalDistKNN {
            irises,
            k,
            next_id,
            pool: ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        let start = self.next_id as usize;
        let end = (start + chunk_size).min(self.irises.len() + 1);
        self.next_id = end as IrisSerialId;

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

                    // Only select if k != 0
                    if self.k >= 1 {
                        neighbors.select_nth_unstable_by(self.k - 1, |lhs, rhs| {
                            fraction_ordering(&lhs.1, &rhs.1)
                        });
                    }

                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit(); // just to make sure
                    neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));

                    let neighbors = neighbors
                        .into_iter()
                        .map(|(i, _)| i as IrisSerialId)
                        .collect::<Vec<_>>();
                    KNNResult {
                        node: i as IrisSerialId,
                        neighbors,
                    }
                })
                .collect::<Vec<_>>()
        })
    }
}

pub struct NaiveMinFHDKNN {
    irises: Vec<IrisCode>,
    k: usize,
    next_id: IrisSerialId,
    pool: ThreadPool,
}

impl KNNEngine for NaiveMinFHDKNN {
    fn init(irises: Vec<IrisCode>, k: usize, next_id: IrisSerialId) -> Self {
        let pool = ThreadPoolBuilder::new().build().unwrap();
        NaiveMinFHDKNN {
            irises,
            k,
            next_id,
            pool,
        }
    }

    fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        let start = self.next_id as usize;
        let end = (start + chunk_size).min(self.irises.len() + 1);
        self.next_id = end as IrisSerialId;

        let irises_with_rotations: Vec<[IrisCode; 31]> = self.pool.install(|| {
            self.irises[(start - 1)..(end - 1)]
                .par_iter()
                .map(|iris| iris.all_rotations().try_into().unwrap())
                .collect()
        });

        self.pool.install(|| {
            (start..end)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|i| {
                    let current_iris = &irises_with_rotations[i - start];
                    let mut neighbors = self
                        .irises
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

                    // Only select if k != 0
                    if self.k >= 1 {
                        neighbors.select_nth_unstable_by(self.k - 1, |lhs, rhs| {
                            match fraction_ordering(&lhs.1, &rhs.1) {
                                Ordering::Equal => lhs.0.cmp(&rhs.0),
                                other => other,
                            }
                        });
                    }

                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit(); // just to make sure
                    neighbors.sort_by(|lhs, rhs| match fraction_ordering(&lhs.1, &rhs.1) {
                        Ordering::Equal => lhs.0.cmp(&rhs.0),
                        other => other,
                    });

                    let neighbors = neighbors
                        .into_iter()
                        .map(|(i, _)| i as IrisSerialId)
                        .collect::<Vec<_>>();
                    KNNResult {
                        node: i as IrisSerialId,
                        neighbors,
                    }
                })
                .collect::<Vec<_>>()
        })
    }
}
