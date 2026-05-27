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

use crate::hawkers::{
    aby3::aby3_store::{DistanceMode, DistanceOps, FhdOps, NhdOps},
    plaintext_deep_id_store::Int4Vector,
};

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
    pub fn distance_mode(&self) -> DistanceMode {
        match self {
            EngineChoice::NaiveFHD | EngineChoice::NaiveNHD => DistanceMode::Simple,
            EngineChoice::NaiveMinFHD | EngineChoice::NaiveMinNHD => DistanceMode::MinRotation,
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
        let distance_mode = which.distance_mode();
        match which {
            EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => {
                Self::Fhd(NaiveKNN::init(irises, k, next_id, distance_mode))
            }
            EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => {
                Self::Nhd(NaiveKNN::init(irises, k, next_id, distance_mode))
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
    distance_mode: DistanceMode,
    pool: ThreadPool,
    _phantom: PhantomData<D>,
}

impl<D: DistanceOps> NaiveKNN<D> {
    pub fn init(
        irises: Vec<IrisCode>,
        k: usize,
        next_id: IrisSerialId,
        distance_mode: DistanceMode,
    ) -> Self {
        NaiveKNN {
            irises,
            k,
            next_id,
            distance_mode,
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
                                D::plaintext_distance(current_iris, other_iris, self.distance_mode),
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

/* ------------------------------ Int4 engine ----------------------------- */

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineChoiceInt4 {
    NaiveInt4Dot,
}

pub struct NaiveKNNInt4 {
    vectors: Vec<Int4Vector>,
    k: usize,
    next_id: IrisSerialId,
    pool: ThreadPool,
}

impl NaiveKNNInt4 {
    pub fn init(vectors: Vec<Int4Vector>, k: usize, next_id: IrisSerialId) -> Self {
        Self {
            vectors,
            k,
            next_id,
            pool: ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    /// Inner-product KNN: larger dot ⇒ closer (matches `PlaintextDeepIDStore::less_than`).
    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        let start = self.next_id as usize;
        let end = (start + chunk_size).min(self.vectors.len() + 1);
        self.next_id = end as IrisSerialId;

        self.pool.install(|| {
            (start..end)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|i| {
                    let current = &self.vectors[i - 1];
                    let mut neighbors = self
                        .vectors
                        .iter()
                        .enumerate()
                        .flat_map(|(j, other)| {
                            (i != j + 1).then_some((j + 1, current.dot(other)))
                        })
                        .collect::<Vec<_>>();

                    if self.k >= 1 {
                        // Larger dot = closer, so order by descending dot.
                        neighbors.select_nth_unstable_by(self.k - 1, |lhs, rhs| {
                            match rhs.1.cmp(&lhs.1) {
                                Ordering::Equal => lhs.0.cmp(&rhs.0),
                                other => other,
                            }
                        });
                    }

                    let mut neighbors = neighbors.drain(0..self.k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit();
                    neighbors.sort_by(|lhs, rhs| match rhs.1.cmp(&lhs.1) {
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

pub enum EngineInt4 {
    Int4Dot(NaiveKNNInt4),
}

impl EngineInt4 {
    pub fn init(
        which: EngineChoiceInt4,
        vectors: Vec<Int4Vector>,
        k: usize,
        next_id: IrisSerialId,
    ) -> Self {
        assert!(k < vectors.len());
        match which {
            EngineChoiceInt4::NaiveInt4Dot => {
                Self::Int4Dot(NaiveKNNInt4::init(vectors, k, next_id))
            }
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult<IrisSerialId>> {
        match self {
            Self::Int4Dot(engine) => engine.compute_chunk(chunk_size),
        }
    }

    pub fn next_id(&self) -> IrisSerialId {
        match self {
            Self::Int4Dot(engine) => engine.next_id,
        }
    }
}

#[cfg(test)]
mod int4_engine_tests {
    use super::*;
    use aes_prng::AesRng;
    use rand::SeedableRng;

    #[test]
    fn int4_knn_returns_top_k_by_descending_dot() {
        let mut rng = AesRng::seed_from_u64(0xCAFE);
        let n = 16;
        let k = 3;
        let vectors: Vec<Int4Vector> = (0..n).map(|_| Int4Vector::random(&mut rng)).collect();

        let mut engine =
            EngineInt4::init(EngineChoiceInt4::NaiveInt4Dot, vectors.clone(), k, 1);
        let results = engine.compute_chunk(n);

        assert_eq!(results.len(), n);
        for KNNResult { node, neighbors } in results {
            // Brute-force expected top-k by descending dot (excluding self).
            let me = &vectors[node as usize - 1];
            let mut dists: Vec<(IrisSerialId, i16)> = (1..=n as IrisSerialId)
                .filter(|j| *j != node)
                .map(|j| (j, me.dot(&vectors[j as usize - 1])))
                .collect();
            dists.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let expected: Vec<IrisSerialId> = dists.into_iter().take(k).map(|(j, _)| j).collect();
            assert_eq!(neighbors, expected, "node {} top-{}", node, k);
        }
    }
}
