use std::{
    cmp::Ordering,
    fs::File,
    io::{BufRead, BufReader},
    marker::PhantomData,
    path::PathBuf,
};

use clap::ValueEnum;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};
use serde::{Deserialize, Serialize};

use crate::hawkers::aby3::aby3_store::{DistanceFn, DistanceOps, FhdOps, NhdOps};

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

impl<V> KNNResult<V> {
    pub fn truncate(&mut self, k: usize) {
        assert!(k <= self.neighbors.len(), "k must be <= neighbors.len()");
        self.neighbors.truncate(k);
        self.neighbors.shrink_to_fit();
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

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineChoice {
    NaiveFHD,
    NaiveMinFHD,
    NaiveNHD,
    NaiveMinNHD,
}

impl EngineChoice {
    pub fn distance_fn(&self) -> DistanceFn {
        match self {
            EngineChoice::NaiveFHD | EngineChoice::NaiveNHD => DistanceFn::Simple,
            EngineChoice::NaiveMinFHD | EngineChoice::NaiveMinNHD => DistanceFn::MinRotation,
        }
    }
}

pub enum Engine {
    Fhd(NaiveKNN<FhdOps>),
    Nhd(NaiveKNN<NhdOps>),
}

impl Engine {
    pub fn init(
        which: EngineChoice,
        irises: Vec<IrisCode>,
        k: usize,
        next_id: IrisSerialId,
    ) -> Self {
        assert!(k < irises.len());
        let distance_fn = which.distance_fn();
        match which {
            EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => {
                Self::Fhd(NaiveKNN::init(irises, k, next_id, distance_fn))
            }
            EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => {
                Self::Nhd(NaiveKNN::init(irises, k, next_id, distance_fn))
            }
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        match self {
            Self::Fhd(engine) => engine.compute_chunk(chunk_size),
            Self::Nhd(engine) => engine.compute_chunk(chunk_size),
        }
    }

    /// Next id to process
    pub fn next_id(&self) -> IrisSerialId {
        match self {
            Self::Fhd(engine) => engine.next_id,
            Self::Nhd(engine) => engine.next_id,
        }
    }
}

pub struct NaiveKNN<D: DistanceOps> {
    irises: Vec<IrisCode>,
    k: usize,
    next_id: IrisSerialId,
    distance_fn: DistanceFn,
    pool: ThreadPool,
    _phantom: PhantomData<D>,
}

impl<D: DistanceOps> NaiveKNN<D> {
    pub fn init(
        irises: Vec<IrisCode>,
        k: usize,
        next_id: IrisSerialId,
        distance_fn: DistanceFn,
    ) -> Self {
        NaiveKNN {
            irises,
            k,
            next_id,
            distance_fn,
            pool: ThreadPoolBuilder::new().build().unwrap(),
            _phantom: PhantomData,
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
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
                            (i != j + 1).then_some((
                                j + 1,
                                D::plaintext_distance(current_iris, other_iris, self.distance_fn),
                            ))
                        })
                        .collect::<Vec<_>>();

                    // Only select if k != 0
                    if self.k >= 1 {
                        neighbors.select_nth_unstable_by(self.k - 1, |lhs, rhs| {
                            match D::plaintext_ordering(&lhs.1, &rhs.1) {
                                Ordering::Equal => lhs.0.cmp(&rhs.0),
                                other => other,
                            }
                        });
                    }

                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit();
                    neighbors.sort_by(|lhs, rhs| match D::plaintext_ordering(&lhs.1, &rhs.1) {
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
