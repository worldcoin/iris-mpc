use std::{
    cmp::Ordering,
    fs::File,
    io::{BufRead, BufReader},
    marker::PhantomData,
    path::PathBuf,
};

use clap::ValueEnum;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId, IrisVectorId};
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
pub struct KNNResult {
    pub node: IrisVectorId,
    pub neighbors: Vec<IrisVectorId>,
}

impl KNNResult {
    pub fn map<F>(self, mut f: F) -> KNNResult
    where
        F: FnMut(IrisVectorId) -> IrisVectorId,
    {
        KNNResult {
            node: f(self.node),
            neighbors: self.neighbors.into_iter().map(f).collect(),
        }
    }

    pub fn truncate(&mut self, k: usize) {
        assert!(k <= self.neighbors.len(), "k must be <= neighbors.len()");
        self.neighbors.truncate(k);
        self.neighbors.shrink_to_fit();
    }
}

/// Reads a `Vec<KNNResult>` from a file, skipping the first line (header).
pub fn read_knn_results_from_file(path: PathBuf) -> std::io::Result<Vec<KNNResult>> {
    #[derive(Deserialize)]
    struct KNNResultU32 {
        node: u32,
        neighbors: Vec<u32>,
    }

    let file = File::open(path)?;
    let mut lines = BufReader::new(file).lines();

    // Skip the header
    lines.next();

    let mut results = Vec::new();
    for line in lines {
        let line = line?;
        let knn_result_u32: KNNResultU32 = serde_json::from_str(&line)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let knn_result = KNNResult {
            node: IrisVectorId::from_serial_id(knn_result_u32.node),
            neighbors: knn_result_u32
                .neighbors
                .into_iter()
                .map(IrisVectorId::from_serial_id)
                .collect(),
        };
        results.push(knn_result);
    }
    Ok(results)
}

/* ---------------------- Generic ideal-KNN abstraction ------------------- */

/// Pluggable distance + ordering used by [`NaiveKNN`].
///
/// Implementations carry whatever runtime state the distance function needs
/// (e.g. iris `DistanceMode`); the generic engine treats both iris-code and
/// Int4 deep-ID vectors uniformly.
pub trait IdealKnn: Send + Sync + Clone + 'static {
    type Vector: Send + Sync + Clone;
    type Distance: Copy + Send + Sync;

    fn distance(&self, a: &Self::Vector, b: &Self::Vector) -> Self::Distance;
    fn order(a: &Self::Distance, b: &Self::Distance) -> Ordering;
}

/// Iris-code distance using `DistanceOps`.
pub struct IrisKnn<D: DistanceOps> {
    distance_mode: DistanceMode,
    _phantom: PhantomData<D>,
}

impl<D: DistanceOps> Clone for IrisKnn<D> {
    fn clone(&self) -> Self {
        Self {
            distance_mode: self.distance_mode,
            _phantom: PhantomData,
        }
    }
}

impl<D: DistanceOps> IrisKnn<D> {
    pub fn new(distance_mode: DistanceMode) -> Self {
        Self {
            distance_mode,
            _phantom: PhantomData,
        }
    }
}

impl<D: DistanceOps + Send + Sync + 'static> IdealKnn for IrisKnn<D> {
    type Vector = IrisCode;
    type Distance = (u16, u16);

    fn distance(&self, a: &IrisCode, b: &IrisCode) -> (u16, u16) {
        D::plaintext_distance(a, b, self.distance_mode)
    }
    fn order(a: &(u16, u16), b: &(u16, u16)) -> Ordering {
        D::plaintext_ordering(a, b)
    }
}

/// Int4 deep-ID inner-product (larger dot = closer).
#[derive(Clone, Copy)]
pub struct Int4DotKnn;

impl IdealKnn for Int4DotKnn {
    type Vector = Int4Vector;
    type Distance = i32;

    fn distance(&self, a: &Int4Vector, b: &Int4Vector) -> i32 {
        a.dot(b)
    }
    fn order(a: &i32, b: &i32) -> Ordering {
        // larger dot is "more similar", so e.g. a > b translates to Ordering::Less
        b.cmp(a)
    }
}

/* ---------------------- Choices kept for public API --------------------- */

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

#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineChoiceInt4 {
    NaiveInt4Dot,
}

/// Unified CLI-facing engine selector spanning both the iris-code engines and
/// the deep-ID Int4 engine. Used by binaries that accept either store kind via
/// a single `--engine-choice` flag; the concrete sub-enum is recovered with
/// [`EngineKind::as_iris`] / [`EngineKind::as_int4`].
#[derive(Clone, Debug, ValueEnum, Copy, Serialize, Deserialize, PartialEq)]
pub enum EngineKind {
    NaiveFHD,
    NaiveMinFHD,
    NaiveNHD,
    NaiveMinNHD,
    NaiveInt4Dot,
}

impl EngineKind {
    /// Whether this selects the deep-ID Int4 engine (vs an iris-code engine).
    pub fn is_int4(&self) -> bool {
        matches!(self, EngineKind::NaiveInt4Dot)
    }

    /// The iris-code [`EngineChoice`], or `None` for the Int4 variant.
    pub fn as_iris(&self) -> Option<EngineChoice> {
        match self {
            EngineKind::NaiveFHD => Some(EngineChoice::NaiveFHD),
            EngineKind::NaiveMinFHD => Some(EngineChoice::NaiveMinFHD),
            EngineKind::NaiveNHD => Some(EngineChoice::NaiveNHD),
            EngineKind::NaiveMinNHD => Some(EngineChoice::NaiveMinNHD),
            EngineKind::NaiveInt4Dot => None,
        }
    }

    /// The deep-ID [`EngineChoiceInt4`], or `None` for an iris variant.
    pub fn as_int4(&self) -> Option<EngineChoiceInt4> {
        match self {
            EngineKind::NaiveInt4Dot => Some(EngineChoiceInt4::NaiveInt4Dot),
            _ => None,
        }
    }
}

/* -------------------------- Generic naive engine ------------------------ */

pub struct NaiveKNN<K: IdealKnn> {
    knn: K,
    vectors: Vec<K::Vector>,
    k: usize,
    next_id: IrisSerialId,
    pool: ThreadPool,
}

impl<K: IdealKnn> NaiveKNN<K> {
    pub fn init(knn: K, vectors: Vec<K::Vector>, k: usize, next_id: IrisSerialId) -> Self {
        // The chunk loop indexes `self.vectors[i - 1]` for `i in next_id..end`,
        // so a starting id of 0 would underflow.
        assert!(next_id >= 1, "next_id must be >= 1 (1-based serial ids)");
        Self {
            knn,
            vectors,
            k,
            next_id,
            pool: ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    pub fn next_id(&self) -> IrisSerialId {
        self.next_id
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        let start = self.next_id as usize;
        let end = (start + chunk_size).min(self.vectors.len() + 1);
        self.next_id = end as IrisSerialId;

        let k = self.k;
        let knn = &self.knn;
        let vectors = &self.vectors;

        self.pool.install(|| {
            (start..end)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|i| {
                    let current = &vectors[i - 1];
                    let mut neighbors = vectors
                        .iter()
                        .enumerate()
                        .flat_map(|(j, other)| {
                            (i != j + 1).then_some((j + 1, knn.distance(current, other)))
                        })
                        .collect::<Vec<_>>();

                    if k >= 1 {
                        neighbors.select_nth_unstable_by(k - 1, |lhs, rhs| {
                            match K::order(&lhs.1, &rhs.1) {
                                Ordering::Equal => lhs.0.cmp(&rhs.0),
                                other => other,
                            }
                        });
                    }

                    let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                    neighbors.shrink_to_fit();
                    neighbors.sort_by(|lhs, rhs| match K::order(&lhs.1, &rhs.1) {
                        Ordering::Equal => lhs.0.cmp(&rhs.0),
                        other => other,
                    });

                    KNNResult {
                        node: IrisVectorId::from_serial_id(i as IrisSerialId),
                        neighbors: neighbors
                            .into_iter()
                            .map(|(j, _)| IrisVectorId::from_serial_id(j as IrisSerialId))
                            .collect(),
                    }
                })
                .collect::<Vec<_>>()
        })
    }
}

/* ----------- Existing `Engine`/`EngineInt4` enums (compat shims) -------- */

pub enum Engine {
    Fhd(NaiveKNN<IrisKnn<FhdOps>>),
    Nhd(NaiveKNN<IrisKnn<NhdOps>>),
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
            EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => Self::Fhd(NaiveKNN::init(
                IrisKnn::new(distance_mode),
                irises,
                k,
                next_id,
            )),
            EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => Self::Nhd(NaiveKNN::init(
                IrisKnn::new(distance_mode),
                irises,
                k,
                next_id,
            )),
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        match self {
            Self::Fhd(engine) => engine.compute_chunk(chunk_size),
            Self::Nhd(engine) => engine.compute_chunk(chunk_size),
        }
    }

    /// Next id to process
    pub fn next_id(&self) -> IrisSerialId {
        match self {
            Self::Fhd(engine) => engine.next_id(),
            Self::Nhd(engine) => engine.next_id(),
        }
    }
}

pub enum EngineInt4 {
    Int4Dot(NaiveKNN<Int4DotKnn>),
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
                Self::Int4Dot(NaiveKNN::init(Int4DotKnn, vectors, k, next_id))
            }
        }
    }

    pub fn compute_chunk(&mut self, chunk_size: usize) -> Vec<KNNResult> {
        match self {
            Self::Int4Dot(engine) => engine.compute_chunk(chunk_size),
        }
    }

    pub fn next_id(&self) -> IrisSerialId {
        match self {
            Self::Int4Dot(engine) => engine.next_id(),
        }
    }
}

#[cfg(test)]
mod engine_kind_tests {
    use super::*;

    #[test]
    fn is_int4_only_for_int4_variant() {
        assert!(!EngineKind::NaiveFHD.is_int4());
        assert!(!EngineKind::NaiveMinFHD.is_int4());
        assert!(!EngineKind::NaiveNHD.is_int4());
        assert!(!EngineKind::NaiveMinNHD.is_int4());
        assert!(EngineKind::NaiveInt4Dot.is_int4());
    }

    #[test]
    fn as_iris_maps_the_four_iris_variants() {
        assert_eq!(EngineKind::NaiveFHD.as_iris(), Some(EngineChoice::NaiveFHD));
        assert_eq!(
            EngineKind::NaiveMinFHD.as_iris(),
            Some(EngineChoice::NaiveMinFHD)
        );
        assert_eq!(EngineKind::NaiveNHD.as_iris(), Some(EngineChoice::NaiveNHD));
        assert_eq!(
            EngineKind::NaiveMinNHD.as_iris(),
            Some(EngineChoice::NaiveMinNHD)
        );
        assert_eq!(EngineKind::NaiveInt4Dot.as_iris(), None);
    }

    #[test]
    fn as_int4_maps_only_the_int4_variant() {
        assert_eq!(
            EngineKind::NaiveInt4Dot.as_int4(),
            Some(EngineChoiceInt4::NaiveInt4Dot)
        );
        assert_eq!(EngineKind::NaiveFHD.as_int4(), None);
        assert_eq!(EngineKind::NaiveNHD.as_int4(), None);
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

        let mut engine = EngineInt4::init(EngineChoiceInt4::NaiveInt4Dot, vectors.clone(), k, 1);
        let results = engine.compute_chunk(n);

        assert_eq!(results.len(), n);
        for KNNResult { node, neighbors } in results {
            // Brute-force expected top-k by descending dot (excluding self).
            let node_serial = node.serial_id();
            let me = &vectors[node_serial as usize - 1];
            let mut dists: Vec<(IrisSerialId, i32)> = (1..=n as IrisSerialId)
                .filter(|j| *j != node_serial)
                .map(|j| (j, me.dot(&vectors[j as usize - 1])))
                .collect();
            dists.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let expected: Vec<IrisVectorId> = dists
                .into_iter()
                .take(k)
                .map(|(j, _)| IrisVectorId::from_serial_id(j))
                .collect();
            assert_eq!(neighbors, expected, "node {} top-{}", node, k);
        }
    }
}
