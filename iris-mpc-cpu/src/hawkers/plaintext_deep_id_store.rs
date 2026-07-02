//! Plaintext `VectorStore` over packed int4 vectors with inner-product distance.
//!
//! Benchmarking/experimentation harness — not a production path and not an MPC
//! mirror. Each vector is 512 signed nibbles in the range `{-8..=7}` packed two
//! per byte. Distance between two vectors is their integer inner product
//! (similarity, not Hamming distance). The store fires a match when the inner
//! product exceeds a configurable threshold.

use crate::{
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    hnsw::{
        metrics::ops_counter::Operation::{CompareDistance, EvaluateDistance},
        vector_store::VectorStoreMut,
        GraphMem, HnswSearcher, VectorStore,
    },
};
use aes_prng::AesRng;
use eyre::{bail, Result};
use iris_mpc_common::{SerialId, VectorId};
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::sync::Arc;
use tracing::debug;

/// Number of int4 elements in each vector.
pub const INT4_DIM: usize = 512;

/// Bytes per packed vector (two int4 elements per byte).
pub const INT4_PACKED_BYTES: usize = INT4_DIM / 2;

/// 512-element vector of signed 4-bit values in `{-8..=7}` packed two per byte
/// using two's-complement nibbles.
///
/// Byte `i` carries element `2*i` in its low nibble and element `2*i+1` in its
/// high nibble. All 16 nibble values are valid: `0x0..=0x7` (0..7) and
/// `0x8..=0xF` (-8..-1).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Int4Vector {
    #[serde(with = "BigArray")]
    pub packed: [u8; INT4_PACKED_BYTES],
}

impl Default for Int4Vector {
    fn default() -> Self {
        Self {
            packed: [0u8; INT4_PACKED_BYTES],
        }
    }
}

pub type Int4StoredVector = Arc<Int4Vector>;
pub type Int4SharedVectors = SharedIrises<Int4StoredVector>;
pub type Int4SharedVectorsRef = SharedIrisesRef<Int4StoredVector>;

impl Int4Vector {
    /// Generate a random `Int4Vector` with each element drawn i.i.d. uniformly
    /// from `{-8..=7}`.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut packed = [0u8; INT4_PACKED_BYTES];
        for byte in packed.iter_mut() {
            let lo = Self::encode_nibble(rng.gen_range(-8..=7));
            let hi = Self::encode_nibble(rng.gen_range(-8..=7));
            *byte = lo | (hi << 4);
        }
        Self { packed }
    }

    /// Decode element at index `i` (0-based). Returns a value in `{-8..=7}`.
    pub fn get(&self, i: usize) -> i8 {
        let byte = self.packed[i / 2];
        let nibble = if i.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        Self::decode_nibble(nibble)
    }

    /// Set element at index `i` (0-based) to `value`, which must lie in
    /// `{-8..=7}`.
    ///
    /// Out-of-range values follow the same masking behavior as
    /// `encode_nibble`: silently masked in release, debug-asserted.
    pub fn set(&mut self, i: usize, value: i8) {
        let byte = &mut self.packed[i / 2];
        let nibble = Self::encode_nibble(value);
        if i.is_multiple_of(2) {
            *byte = (*byte & 0xF0) | nibble;
        } else {
            *byte = (*byte & 0x0F) | (nibble << 4);
        }
    }

    /// Integer inner product of two int4 vectors.
    ///
    /// Decodes nibbles on the fly and accumulates element-wise products in
    /// `i32`. The tightest bound for the `{-8..=7}` domain is
    /// `512 * 8 * 8 = 32_768`, which exceeds `i16::MAX = 32_767` but fits
    /// comfortably in `i32`.
    pub fn dot(&self, other: &Self) -> i32 {
        let mut acc: i32 = 0;
        for (a, b) in self.packed.iter().zip(other.packed.iter()) {
            let a_lo = i32::from(Self::decode_nibble(*a & 0x0F));
            let a_hi = i32::from(Self::decode_nibble(*a >> 4));
            let b_lo = i32::from(Self::decode_nibble(*b & 0x0F));
            let b_hi = i32::from(Self::decode_nibble(*b >> 4));
            acc += a_lo * b_lo + a_hi * b_hi;
        }
        acc
    }

    /// Encode a value in `{-8..=7}` as a 4-bit two's-complement nibble.
    ///
    /// Values outside `{-8..=7}` are masked to a nibble silently in release;
    /// `+8` aliases to `-8` (both encode to `0x08`). A debug assertion catches
    /// out-of-range inputs to surface misuse early.
    #[inline]
    fn encode_nibble(value: i8) -> u8 {
        debug_assert!(
            (-8..=7).contains(&value),
            "Int4Vector value {value} outside supported domain {{-8..=7}}",
        );
        (value as u8) & 0x0F
    }

    /// Decode a 4-bit two's-complement nibble to a signed `i8`.
    #[inline]
    fn decode_nibble(nibble: u8) -> i8 {
        let n = nibble & 0x0F;
        if n & 0x08 != 0 {
            (n as i8) | !0x0F_i8 // sign-extend
        } else {
            n as i8
        }
    }
}

/// Whether a distance counts as a match: the inner product (a similarity, not
/// a Hamming distance) must strictly exceed `threshold`.
///
/// Shared by [`PlaintextDeepIDStore`] and [`SharedPlaintextDeepIDStore`] so the
/// match rule lives in exactly one place.
#[inline]
fn dot_is_match(distance: i32, threshold: i32) -> bool {
    distance > threshold
}

/// Whether `d1` is closer than `d2` under similarity ordering: a larger dot
/// means "closer", so this holds iff `d1 > d2`.
///
/// Shared by both stores so the ordering rule lives in exactly one place.
#[inline]
fn dot_is_closer(d1: i32, d2: i32) -> bool {
    d1 > d2
}

/// Single-threaded plaintext store over packed int4 vectors.
///
/// Distance is the integer inner product; a pair is a "match" iff the dot
/// exceeds `threshold`. Mirrors the shape of [`PlaintextStore`] but with
/// simpler distance and no dependency on iris codes.
///
/// [`PlaintextStore`]: super::plaintext_store::PlaintextStore
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct PlaintextDeepIDStore {
    pub storage: Int4SharedVectors,
    pub threshold: i32,
}

impl PlaintextDeepIDStore {
    /// New empty store with the given match threshold.
    pub fn new(threshold: i32) -> Self {
        Self::with_storage(threshold, Int4SharedVectors::default())
    }

    pub fn with_storage(threshold: i32, storage: Int4SharedVectors) -> Self {
        Self { storage, threshold }
    }

    pub fn len(&self) -> usize {
        self.storage.db_size()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.db_size() == 0
    }

    pub fn insert_with_id(&mut self, id: VectorId, v: Arc<Int4Vector>) -> VectorId {
        self.storage.insert(id, v)
    }

    /// Generate a store of `store_size` random vectors with the given match
    /// threshold, keyed by sequential 0-indexed `VectorId`s.
    pub fn new_random<R: RngCore>(rng: &mut R, store_size: usize, threshold: i32) -> Self {
        let mut store = Self::new(threshold);
        for idx in 0..store_size {
            let id = VectorId::from_0_index(idx as u32);
            store.insert_with_id(id, Arc::new(Int4Vector::random(rng)));
        }
        store
    }

    /// Build an HNSW graph over the first `graph_size` entries of this store
    /// (sorted by serial id) using the given searcher and randomness.
    ///
    /// Mirrors [`PlaintextStore::generate_graph`].
    ///
    /// [`PlaintextStore::generate_graph`]: super::plaintext_store::PlaintextStore::generate_graph
    pub async fn generate_graph<R: RngCore + Clone + CryptoRng>(
        &mut self,
        rng: &mut R,
        graph_size: usize,
        searcher: &HnswSearcher,
    ) -> Result<GraphMem> {
        let mut graph: GraphMem = GraphMem::new();
        let mut rng = AesRng::from_rng(rng.clone())?;

        if graph_size > self.len() {
            bail!("Cannot generate graph larger than underlying vector store");
        }

        // sort in order to ensure deterministic behavior
        let mut serial_ids: Vec<_> = self.storage.get_sorted_serial_ids();
        serial_ids.truncate(graph_size);

        for serial_id in serial_ids {
            let query = self
                .storage
                .get_vector_by_serial_id(serial_id)
                .ok_or_else(|| {
                    eyre::eyre!(
                        "Vector for serial {} not found while generating graph",
                        serial_id
                    )
                })?
                .clone();
            let query_id = self
                .storage
                .get_current_version(serial_id)
                .map(|version| VectorId::new(serial_id, version))
                .unwrap_or_else(|| VectorId::from_serial_id(serial_id));
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert(self, &graph, &query, insertion_layer)
                .await?;
            searcher
                .insert_from_search_results(self, &mut graph, query_id, neighbors, update_ep)
                .await?;
        }

        Ok(graph)
    }
}

impl VectorStore for PlaintextDeepIDStore {
    type QueryRef = Arc<Int4Vector>;
    type DistanceRef = i32;

    async fn vectors_as_queries(&mut self, vectors: Vec<VectorId>) -> Result<Vec<Self::QueryRef>> {
        let queries = vectors
            .into_iter()
            .map(|id| {
                self.storage.get_vector(&id).cloned().ok_or_else(|| {
                    eyre::eyre!("Vector ID not found in store for serial {}", id.serial_id())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(queries)
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &VectorId,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let stored = self.storage.get_vector(vector).ok_or_else(|| {
            eyre::eyre!(
                "Vector ID not found in store for serial {}",
                vector.serial_id()
            )
        })?;
        Ok(query.dot(stored))
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(dot_is_match(*distance, self.threshold))
    }

    async fn less_than(&mut self, d1: &Self::DistanceRef, d2: &Self::DistanceRef) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(dot_is_closer(*d1, *d2))
    }

    // Default implementation + metrics, mirroring PlaintextStore.
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(distances.len());
        for (d1, d2) in distances {
            results.push(self.less_than(d1, d2).await?);
        }
        metrics::counter!("less_than").increment(distances.len() as u64);
        Ok(results)
    }

    async fn serials_to_vector_ids(&self, serial_ids: &[SerialId]) -> Vec<Option<VectorId>> {
        serial_ids
            .iter()
            .map(|&serial_id| {
                self.storage
                    .get_current_version(serial_id)
                    .map(|version| VectorId::new(serial_id, version))
            })
            .collect()
    }
}

impl VectorStoreMut for PlaintextDeepIDStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> VectorId {
        self.storage.append(query.clone())
    }

    async fn insert_at(
        &mut self,
        vector_ref: &VectorId,
        query: &Self::QueryRef,
    ) -> Result<VectorId> {
        Ok(self.storage.insert(*vector_ref, query.clone()))
    }
}

/// `PlaintextDeepIDStore` with synchronization primitives for multithreaded use.
#[derive(Debug, Clone)]
pub struct SharedPlaintextDeepIDStore {
    pub storage: Int4SharedVectorsRef,
    pub threshold: i32,
}

impl SharedPlaintextDeepIDStore {
    pub fn new(threshold: i32) -> Self {
        Self {
            storage: Int4SharedVectors::default().to_arc(),
            threshold,
        }
    }

    pub async fn len(&self) -> usize {
        self.storage.read().await.db_size()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

impl From<PlaintextDeepIDStore> for SharedPlaintextDeepIDStore {
    fn from(value: PlaintextDeepIDStore) -> Self {
        Self {
            storage: value.storage.to_arc(),
            threshold: value.threshold,
        }
    }
}

impl VectorStore for SharedPlaintextDeepIDStore {
    type QueryRef = Arc<Int4Vector>;
    type DistanceRef = i32;

    async fn vectors_as_queries(&mut self, vectors: Vec<VectorId>) -> Result<Vec<Self::QueryRef>> {
        let store = self.storage.read().await;
        let queries = vectors
            .into_iter()
            .map(|id| {
                store.get_vector(&id).cloned().ok_or_else(|| {
                    eyre::eyre!("Vector ID not found in store for serial {}", id.serial_id())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(queries)
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &VectorId,
    ) -> Result<Self::DistanceRef> {
        let distances = self.eval_distance_batch(query, &[*vector]).await?;
        Ok(distances[0])
    }

    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[VectorId],
    ) -> Result<Vec<Self::DistanceRef>> {
        debug!(event_type = EvaluateDistance.id());
        // Snapshot Arc handles under the read lock, drop the guard, then
        // compute dots without blocking concurrent writers.
        let stored = {
            let store = self.storage.read().await;
            vectors
                .iter()
                .map(|v| {
                    store.get_vector(v).cloned().ok_or_else(|| {
                        eyre::eyre!("Vector ID not found in store for serial {}", v.serial_id())
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };
        Ok(stored.into_iter().map(|s| query.dot(&s)).collect())
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(dot_is_match(*distance, self.threshold))
    }

    async fn less_than(&mut self, d1: &Self::DistanceRef, d2: &Self::DistanceRef) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(dot_is_closer(*d1, *d2))
    }

    // Default implementation + metrics, mirroring SharedPlaintextStore.
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(distances.len());
        for (d1, d2) in distances {
            results.push(self.less_than(d1, d2).await?);
        }
        metrics::counter!("less_than").increment(distances.len() as u64);
        Ok(results)
    }

    async fn serials_to_vector_ids(&self, serial_ids: &[SerialId]) -> Vec<Option<VectorId>> {
        let storage = self.storage.read().await;
        serial_ids
            .iter()
            .map(|&serial_id| {
                storage
                    .get_current_version(serial_id)
                    .map(|version| VectorId::new(serial_id, version))
            })
            .collect()
    }
}

impl VectorStoreMut for SharedPlaintextDeepIDStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> VectorId {
        self.storage.append(query).await
    }

    async fn insert_at(
        &mut self,
        vector_ref: &VectorId,
        query: &Self::QueryRef,
    ) -> Result<VectorId> {
        Ok(self.storage.insert(*vector_ref, query).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::SortedNeighborhood;
    use tokio::task::JoinSet;

    #[test]
    fn test_int4_pack_roundtrip() {
        let mut rng = AesRng::seed_from_u64(0x1234_5678_9abc_def0);
        let v = Int4Vector::random(&mut rng);
        for i in 0..INT4_DIM {
            let x = v.get(i);
            assert!(
                (-8..=7).contains(&x),
                "element {i} = {x} is outside {{-8..=7}}",
            );
        }

        // Hand-constructed vector: alternating +7 and -7
        let mut packed = [0u8; INT4_PACKED_BYTES];
        for byte in packed.iter_mut() {
            // low nibble = +7 (0x07), high nibble = -7 (0x09)
            *byte = 0x07 | (0x09 << 4);
        }
        let v = Int4Vector { packed };
        for i in 0..INT4_DIM {
            let expected = if i.is_multiple_of(2) { 7 } else { -7 };
            assert_eq!(v.get(i), expected, "mismatch at element {i}");
        }
    }

    #[test]
    fn test_int4_set_roundtrip() {
        // Set every element individually, then read it back via get().
        let mut v = Int4Vector::default();
        for i in 0..INT4_DIM {
            // Sweep the full domain {-8..=7} across indices.
            let value = ((i % 16) as i8) - 8;
            v.set(i, value);
        }
        for i in 0..INT4_DIM {
            let expected = ((i % 16) as i8) - 8;
            assert_eq!(v.get(i), expected, "mismatch at element {i}");
        }
    }

    #[test]
    fn test_int4_set_overwrites_without_disturbing_neighbor() {
        // Writing the low nibble must not clobber the high nibble and vice versa.
        let mut v = Int4Vector::default();
        v.set(0, 7); // low nibble of byte 0
        v.set(1, -8); // high nibble of byte 0
        assert_eq!(v.get(0), 7);
        assert_eq!(v.get(1), -8);

        // Overwrite each independently.
        v.set(0, -1);
        assert_eq!(v.get(0), -1);
        assert_eq!(
            v.get(1),
            -8,
            "neighbor nibble disturbed by low-nibble write"
        );

        v.set(1, 3);
        assert_eq!(v.get(1), 3);
        assert_eq!(
            v.get(0),
            -1,
            "neighbor nibble disturbed by high-nibble write"
        );
    }

    fn make_vec_with(value: i8) -> Int4Vector {
        let mut v = Int4Vector::default();
        for i in 0..INT4_DIM {
            v.set(i, value);
        }
        v
    }

    #[test]
    fn test_dot_known_values() {
        let zero = Int4Vector::default();
        assert_eq!(zero.dot(&zero), 0);

        let all_plus_7 = make_vec_with(7);
        let all_minus_7 = make_vec_with(-7);

        // 512 * 7 * 7 = 25_088
        assert_eq!(all_plus_7.dot(&all_plus_7), 25_088);
        assert_eq!(all_minus_7.dot(&all_minus_7), 25_088);
        assert_eq!(all_plus_7.dot(&all_minus_7), -25_088);

        // Half +7, half -7 in one vector; dotted against all +7 → 0
        let mut split = Int4Vector::default();
        for i in 0..INT4_DIM {
            // first half of elements: +7; second half: -7
            let v = if i < INT4_DIM / 2 { 7_i8 } else { -7_i8 };
            split.set(i, v);
        }
        assert_eq!(split.dot(&all_plus_7), 0);
    }

    #[test]
    fn test_new_random_populates_store() {
        let mut rng = AesRng::seed_from_u64(0xABCD_1234);
        let store = PlaintextDeepIDStore::new_random(&mut rng, 10, 5_000);

        assert_eq!(store.len(), 10);
        assert_eq!(store.threshold, 5_000);

        // Every vector is addressable by its 0-indexed id and stays in domain.
        for i in 0..10 {
            let id = VectorId::from_0_index(i as u32);
            let v = store.storage.get_vector(&id).expect("vector present");
            for j in 0..INT4_DIM {
                assert!((-8..=7).contains(&v.get(j)));
            }
        }
    }

    /// Build a store of `n` random vectors with the given match threshold,
    /// returning the store plus the `(id, vector)` pairs in 0-indexed id order.
    ///
    /// Advances `rng` exactly as [`PlaintextDeepIDStore::new_random`] does, so
    /// callers can keep using it afterward (e.g. to build a graph) and still get
    /// deterministic results.
    fn build_store_with_random_vectors<R: RngCore>(
        rng: &mut R,
        n: usize,
        threshold: i32,
    ) -> (PlaintextDeepIDStore, Vec<(VectorId, Arc<Int4Vector>)>) {
        let store = PlaintextDeepIDStore::new_random(rng, n, threshold);
        let ids = (0..n)
            .map(|i| {
                let id = VectorId::from_0_index(i as u32);
                let v = store
                    .storage
                    .get_vector(&id)
                    .expect("vector present")
                    .clone();
                (id, v)
            })
            .collect();
        (store, ids)
    }

    /// Flip a single nibble of `v` by adding `delta` (clamped to {-8..=7}).
    fn perturb(v: &Int4Vector, index: usize, delta: i8) -> Int4Vector {
        let mut out = v.clone();
        let new = (out.get(index) + delta).clamp(-8, 7);
        out.set(index, new);
        out
    }

    #[tokio::test]
    async fn test_eval_distance_self_matches_dot() {
        let mut rng = AesRng::seed_from_u64(0xCAFEBABE);
        let (mut store, ids) = build_store_with_random_vectors(&mut rng, 8, 12_000);

        // eval_distance(self, self) should equal vector.dot(vector)
        for (id, v) in &ids {
            let q = Arc::clone(v);
            let d = store.eval_distance(&q, id).await.unwrap();
            assert_eq!(d, v.dot(v));
        }
    }

    #[tokio::test]
    async fn test_less_than_antisymmetric() {
        let mut rng = AesRng::seed_from_u64(0xCAFEBABE);
        let (mut store, ids) = build_store_with_random_vectors(&mut rng, 8, 12_000);

        let q0 = Arc::clone(&ids[0].1);
        let d_self = store.eval_distance(&q0, &ids[0].0).await.unwrap();
        let d_other = store.eval_distance(&q0, &ids[1].0).await.unwrap();
        if d_self != d_other {
            let ab = store.less_than(&d_self, &d_other).await.unwrap();
            let ba = store.less_than(&d_other, &d_self).await.unwrap();
            assert_ne!(
                ab, ba,
                "less_than must be antisymmetric for distinct values"
            );
        }
    }

    #[tokio::test]
    async fn test_is_match_high_magnitude_self_dot() {
        let mut store = PlaintextDeepIDStore::new(12_000);

        // is_match: identity dot of a high-magnitude vector exceeds the threshold
        let high = Arc::new(make_vec_with(7));
        let id_high = VectorStoreMut::insert(&mut store, &high).await;
        let q_high = Arc::clone(&high);
        let d_high = store.eval_distance(&q_high, &id_high).await.unwrap();
        assert!(
            store.is_match(&d_high).await.unwrap(),
            "self-dot {d_high} should exceed threshold 12_000",
        );
    }

    #[tokio::test]
    async fn test_is_match_zero_vector_does_not_match() {
        let mut store = PlaintextDeepIDStore::new(12_000);

        // is_match: zero vector against itself does NOT match
        let zero = Arc::new(Int4Vector::default());
        let id_zero = VectorStoreMut::insert(&mut store, &zero).await;
        let q_zero = Arc::clone(&zero);
        let d_zero = store.eval_distance(&q_zero, &id_zero).await.unwrap();
        assert!(!store.is_match(&d_zero).await.unwrap());
    }

    #[tokio::test]
    async fn test_plaintext_int4_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0xFEEDFACE);

        // Threshold sits well above the random-pair dot (~0 ± hundreds) and well
        // below a random vector's self-dot (~512 * 21.5 = 11_008 on average).
        let (mut store, ids) = build_store_with_random_vectors(&mut rng, 64, 5_000);

        // Remember one inserted vector to perturb into a query later.
        let (anchor_id, anchor_vec) = ids[7].clone();

        // Build the HNSW graph.
        let searcher = HnswSearcher::new_with_test_parameters();
        let graph = store
            .generate_graph(&mut rng, 64, &searcher)
            .await
            .expect("graph generation succeeds");

        // Query = anchor with one element perturbed by 1.
        let query = Arc::new(perturb(&anchor_vec, 17, 1));
        let results: SortedNeighborhood<_> = searcher
            .search(&mut store, &graph, &query, 5)
            .await
            .expect("search succeeds");
        let results = results.as_vec_ref();

        let neighbor_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            neighbor_ids.contains(&anchor_id),
            "anchor {anchor_id} not in top-5 neighbors {neighbor_ids:?}",
        );

        // The anchor's distance must be a match.
        let (_, anchor_distance) = results
            .iter()
            .find(|(id, _)| *id == anchor_id)
            .expect("anchor present");
        assert!(
            store.is_match(anchor_distance).await.unwrap(),
            "perturbed-anchor dot {anchor_distance} should exceed threshold 5_000",
        );
    }

    #[tokio::test]
    async fn test_parallel_plaintext_int4_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0x5EED5EED);
        let (mut store, originals) = build_store_with_random_vectors(&mut rng, 32, 0);

        let searcher = HnswSearcher::new_with_test_parameters();
        let graph = store
            .generate_graph(&mut rng, 32, &searcher)
            .await
            .expect("graph generation succeeds");

        // Convert to the shared store and search the first 8 originals in parallel.
        let shared: SharedPlaintextDeepIDStore = store.into();
        let searcher = Arc::new(searcher);
        let graph = Arc::new(graph);

        let mut join_set = JoinSet::new();
        for (expected_id, vec) in originals.into_iter().take(8) {
            let mut shared = shared.clone();
            let searcher = Arc::clone(&searcher);
            let graph = Arc::clone(&graph);
            join_set.spawn(async move {
                let query = Arc::new((*vec).clone());
                let results: SortedNeighborhood<_> = searcher
                    .search(&mut shared, &graph, &query, 1)
                    .await
                    .expect("search succeeds");
                let pairs = results.as_vec_ref();
                assert!(!pairs.is_empty(), "expected at least one result");
                // Self-dot of any random vector is a sum of squares ≥ 0.
                let (top_id, top_dist) = &pairs[0];
                assert_eq!(*top_id, expected_id);
                assert!(*top_dist >= 0);
            });
        }

        while let Some(res) = join_set.join_next().await {
            res.expect("task panicked");
        }
    }
}
