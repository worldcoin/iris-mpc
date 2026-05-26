//! Plaintext `VectorStore` over packed int4 vectors with inner-product distance.
//!
//! Benchmarking/experimentation harness — not a production path and not an MPC
//! mirror. Each vector is 512 signed nibbles in the range `{-7..=7}` packed two
//! per byte. Distance between two vectors is their integer inner product
//! (similarity, not Hamming distance). The store fires a match when the inner
//! product exceeds a configurable threshold.

use crate::{
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    hnsw::{
        metrics::ops_counter::Operation::{CompareDistance, EvaluateDistance},
        vector_store::VectorStoreMut,
        GraphMem, HnswSearcher, SortedNeighborhood, VectorStore,
    },
};
use aes_prng::AesRng;
use eyre::{bail, Result};
use iris_mpc_common::vector_id::VectorId;
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::sync::Arc;
use tracing::debug;

/// Number of int4 elements in each vector.
pub const INT4_DIM: usize = 512;

/// Bytes per packed vector (two int4 elements per byte).
pub const INT4_PACKED_BYTES: usize = INT4_DIM / 2;

/// 512-element vector of signed 4-bit values in `{-7..=7}` packed two per byte
/// using two's-complement nibbles.
///
/// Byte `i` carries element `2*i` in its low nibble and element `2*i+1` in its
/// high nibble. Valid nibble values are `0x0..=0x7` (positive 0..7) and
/// `0x9..=0xF` (negative -7..-1). The nibble `0x8` (-8) is outside the
/// supported domain — `dot`'s i16 bound assumes `{-7..=7}` and is no longer
/// safe if it appears.
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
    /// from `{-7..=7}`.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut packed = [0u8; INT4_PACKED_BYTES];
        for byte in packed.iter_mut() {
            let lo = Self::encode_nibble(rng.gen_range(-7..=7));
            let hi = Self::encode_nibble(rng.gen_range(-7..=7));
            *byte = lo | (hi << 4);
        }
        Self { packed }
    }

    /// Decode element at index `i` (0-based).
    ///
    /// Returns a value in `{-8..=7}`. Vectors produced by [`Int4Vector::random`]
    /// only ever yield values in `{-7..=7}`.
    pub fn get(&self, i: usize) -> i8 {
        let byte = self.packed[i / 2];
        let nibble = if i.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        Self::decode_nibble(nibble)
    }

    /// Integer inner product of two int4 vectors.
    ///
    /// Decodes nibbles on the fly and accumulates element-wise products in
    /// `i16`. The tightest bound is `512 * 7 * 7 = 25_088 < i16::MAX = 32_767`,
    /// which holds for any vector whose nibbles all decode within `{-7..=7}`.
    /// Vectors containing the out-of-domain nibble `0x8` (-8) can overflow.
    pub fn dot(&self, other: &Self) -> i16 {
        let mut acc: i16 = 0;
        for (a, b) in self.packed.iter().zip(other.packed.iter()) {
            let a_lo = i16::from(Self::decode_nibble(*a & 0x0F));
            let a_hi = i16::from(Self::decode_nibble(*a >> 4));
            let b_lo = i16::from(Self::decode_nibble(*b & 0x0F));
            let b_hi = i16::from(Self::decode_nibble(*b >> 4));
            acc += a_lo * b_lo + a_hi * b_hi;
        }
        acc
    }

    /// Encode a value in `{-7..=7}` as a 4-bit two's-complement nibble.
    ///
    /// Values outside `{-7..=7}` are masked to a nibble silently in release;
    /// `+8` aliases to `-8` (both encode to `0x08`). A debug assertion catches
    /// out-of-range inputs to surface misuse early.
    #[inline]
    fn encode_nibble(value: i8) -> u8 {
        debug_assert!(
            (-7..=7).contains(&value),
            "Int4Vector value {value} outside supported domain {{-7..=7}}",
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
    pub threshold: i16,
}

impl PlaintextDeepIDStore {
    /// New empty store with the given match threshold.
    pub fn new(threshold: i16) -> Self {
        Self::with_storage(threshold, Int4SharedVectors::default())
    }

    pub fn with_storage(threshold: i16, storage: Int4SharedVectors) -> Self {
        Self { storage, threshold }
    }

    pub fn len(&self) -> usize {
        self.storage.db_size()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.db_size() == 0
    }

    pub fn prepare_query(&self, v: Int4Vector) -> Arc<Int4Vector> {
        Arc::new(v)
    }

    pub fn insert_with_id(&mut self, id: VectorId, v: Arc<Int4Vector>) -> VectorId {
        self.storage.insert(id, v)
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
    ) -> Result<GraphMem<<Self as VectorStore>::VectorRef>> {
        let mut graph = GraphMem::new();
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
                .unwrap()
                .clone();
            let query_id = self
                .storage
                .get_current_version(serial_id)
                .map(|version| VectorId::new(serial_id, version))
                .unwrap_or_else(|| VectorId::from_serial_id(serial_id));
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert::<_, SortedNeighborhood<_>>(self, &graph, &query, insertion_layer)
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
    type VectorRef = VectorId;
    type DistanceRef = i16;

    async fn vectors_as_queries(
        &mut self,
        vectors: Vec<Self::VectorRef>,
    ) -> Result<Vec<Self::QueryRef>> {
        Ok(vectors
            .iter()
            .map(|id| self.storage.get_vector(id).unwrap().clone())
            .collect())
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
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
        // dot > threshold means "similar enough" (similarity, not distance).
        Ok(*distance > self.threshold)
    }

    async fn less_than(&mut self, d1: &Self::DistanceRef, d2: &Self::DistanceRef) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        // Larger dot = closer, so "d1 closer than d2" iff dot1 > dot2.
        Ok(d1 > d2)
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

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        vectors.retain(|v| self.storage.contains(v));
        vectors
    }
}

impl VectorStoreMut for PlaintextDeepIDStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(query.clone())
    }

    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        Ok(self.storage.insert(*vector_ref, query.clone()))
    }
}

/// `PlaintextDeepIDStore` with synchronization primitives for multithreaded use.
#[derive(Debug, Clone)]
pub struct SharedPlaintextDeepIDStore {
    pub storage: Int4SharedVectorsRef,
    pub threshold: i16,
}

impl SharedPlaintextDeepIDStore {
    pub fn new(threshold: i16) -> Self {
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

    pub fn prepare_query(&self, v: Int4Vector) -> Arc<Int4Vector> {
        Arc::new(v)
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
    type VectorRef = VectorId;
    type DistanceRef = i16;

    async fn vectors_as_queries(
        &mut self,
        vectors: Vec<Self::VectorRef>,
    ) -> Result<Vec<Self::QueryRef>> {
        let store = self.storage.read().await;
        Ok(vectors
            .iter()
            .map(|id| store.get_vector(id).unwrap().clone())
            .collect())
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        let distances = self.eval_distance_batch(query, &[*vector]).await?;
        Ok(distances[0])
    }

    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
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
        Ok(*distance > self.threshold)
    }

    async fn less_than(&mut self, d1: &Self::DistanceRef, d2: &Self::DistanceRef) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(d1 > d2)
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

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        let store = self.storage.read().await;
        vectors.retain(|v| store.contains(v));
        vectors
    }
}

impl VectorStoreMut for SharedPlaintextDeepIDStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(query).await
    }

    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        Ok(self.storage.insert(*vector_ref, query).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinSet;

    #[test]
    fn test_int4_pack_roundtrip() {
        let mut rng = AesRng::seed_from_u64(0x1234_5678_9abc_def0);
        let v = Int4Vector::random(&mut rng);
        for i in 0..INT4_DIM {
            let x = v.get(i);
            assert!(
                (-7..=7).contains(&x),
                "element {i} = {x} is outside {{-7..=7}}",
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

    fn make_vec_with(value: i8) -> Int4Vector {
        let mut packed = [0u8; INT4_PACKED_BYTES];
        let n = Int4Vector::encode_nibble(value);
        let byte = n | (n << 4);
        for b in packed.iter_mut() {
            *b = byte;
        }
        Int4Vector { packed }
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
        for i in 0..INT4_PACKED_BYTES {
            // first 128 bytes: both nibbles +7; next 128 bytes: both nibbles -7
            let v = if i < INT4_PACKED_BYTES / 2 {
                7_i8
            } else {
                -7_i8
            };
            let n = Int4Vector::encode_nibble(v);
            split.packed[i] = n | (n << 4);
        }
        assert_eq!(split.dot(&all_plus_7), 0);
    }

    /// Build a store with `n` random vectors and return the store plus the list
    /// of `(id, vector)` pairs that were inserted, in insertion order.
    async fn build_store_with_random_vectors(
        threshold: i16,
        n: usize,
        seed: u64,
    ) -> (PlaintextDeepIDStore, Vec<(VectorId, Arc<Int4Vector>)>) {
        let mut rng = AesRng::seed_from_u64(seed);
        let mut store = PlaintextDeepIDStore::new(threshold);
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let v = Arc::new(Int4Vector::random(&mut rng));
            let id = VectorStoreMut::insert(&mut store, &v).await;
            ids.push((id, v));
        }
        (store, ids)
    }

    /// Flip a single nibble of `v` by adding `delta` (clamped to {-7..=7}).
    fn perturb(v: &Int4Vector, index: usize, delta: i8) -> Int4Vector {
        let mut out = v.clone();
        let cur = out.get(index);
        let new = (cur + delta).clamp(-7, 7);
        let byte = out.packed[index / 2];
        let n = Int4Vector::encode_nibble(new);
        out.packed[index / 2] = if index.is_multiple_of(2) {
            (byte & 0xF0) | n
        } else {
            (byte & 0x0F) | (n << 4)
        };
        out
    }

    #[tokio::test]
    async fn test_eval_distance_self_matches_dot() {
        let (mut store, ids) = build_store_with_random_vectors(12_000, 8, 0xCAFEBABE).await;

        // eval_distance(self, self) should equal vector.dot(vector)
        for (id, v) in &ids {
            let q = store.prepare_query((**v).clone());
            let d = store.eval_distance(&q, id).await.unwrap();
            assert_eq!(d, v.dot(v));
        }
    }

    #[tokio::test]
    async fn test_less_than_antisymmetric() {
        let (mut store, ids) = build_store_with_random_vectors(12_000, 8, 0xCAFEBABE).await;

        let q0 = store.prepare_query((*ids[0].1).clone());
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
        let q_high = store.prepare_query((*high).clone());
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
        let q_zero = store.prepare_query((*zero).clone());
        let d_zero = store.eval_distance(&q_zero, &id_zero).await.unwrap();
        assert!(!store.is_match(&d_zero).await.unwrap());
    }

    #[tokio::test]
    async fn test_plaintext_int4_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0xFEEDFACE);

        // Threshold sits well above the random-pair dot (~0 ± hundreds) and well
        // below a random vector's self-dot (~512 * 18.67 ≈ 9_560 on average).
        let mut store = PlaintextDeepIDStore::new(/* threshold */ 5_000);

        // Insert 64 random vectors and remember one to perturb later.
        let mut anchor: Option<(VectorId, Arc<Int4Vector>)> = None;
        for i in 0..64 {
            let v = Arc::new(Int4Vector::random(&mut rng));
            let id = VectorStoreMut::insert(&mut store, &v).await;
            if i == 7 {
                anchor = Some((id, v));
            }
        }
        let (anchor_id, anchor_vec) = anchor.expect("anchor was set");

        // Build the HNSW graph.
        let searcher = HnswSearcher::new_with_test_parameters();
        let graph = store
            .generate_graph(&mut rng, /* graph_size */ 64, &searcher)
            .await
            .expect("graph generation succeeds");

        // Query = anchor with one element perturbed by 1.
        let query = store.prepare_query(perturb(&anchor_vec, 17, 1));
        let results: SortedNeighborhood<_> = searcher
            .search(&mut store, &graph, &query, /* k */ 5)
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
        let mut store = PlaintextDeepIDStore::new(/* threshold */ 0);

        // Insert 32 random vectors, keep all of them as candidate queries.
        let mut originals: Vec<(VectorId, Arc<Int4Vector>)> = Vec::new();
        for _ in 0..32 {
            let v = Arc::new(Int4Vector::random(&mut rng));
            let id = VectorStoreMut::insert(&mut store, &v).await;
            originals.push((id, v));
        }

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
                    .search(&mut shared, &graph, &query, /* k */ 1)
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
