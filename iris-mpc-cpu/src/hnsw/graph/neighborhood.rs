//! Implementation of sorted graph neighborhoods for an HNSW hierarchical graph;
//! based on the `FurthestQueue` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::hnsw::{
    sorting::{
        batcher::partial_batcher_network, quickselect::run_quickselect_with_store,
        quicksort::apply_quicksort, swap_network::apply_swap_network,
    },
    VectorStore,
};
use eyre::Result;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};
use tracing::debug;

pub trait NeighborhoodV2:
    Neighborhood<
    Vector = <<Self as NeighborhoodV2>::V as VectorStore>::VectorRef,
    Distance = <<Self as NeighborhoodV2>::V as VectorStore>::DistanceRef,
>
{
    type V: VectorStore;
}

pub trait NeighborhoodV<V: VectorStore>:
    Neighborhood<Vector = V::VectorRef, Distance = V::DistanceRef>
{
}

/// Trait that captures the requirements for an HNSW candidate list container/data-structure.
/// Implementers should ensure the following post-condition for calls to `trim` methods:
/// - The elements in the container are the smallest `container.length` ones among all insertions.
/// - The last element in the container is the largest one.
#[allow(async_fn_in_trait)]
pub trait Neighborhood: Clone {
    type Vector: Clone;
    type Distance: Clone;

    fn new() -> Self;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn clear(&mut self);

    fn iter(&self) -> impl Iterator<Item = &(Self::Vector, Self::Distance)>;

    fn edge_ids(&self) -> Vec<Self::Vector> {
        self.iter().map(|(v, _)| v.clone()).collect::<Vec<_>>()
    }

    /// Inserts a batch of elements into the neighborhood and applies the necessary
    /// changes to ensure the neighborhood invariant holds
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs
    async fn insert_batch_and_trim<V>(
        &mut self,
        store: &mut V,
        vals: &[(Self::Vector, Self::Distance)],
        k: Option<usize>,
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>;

    /// Apply the invariant-maintaining algorithm
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs.
    async fn trim<V>(&mut self, store: &mut V, k: Option<usize>) -> Result<()>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>,
    {
        self.insert_batch_and_trim(store, &[], k).await
    }

    /// Inserts a single element into the neighborhood.
    /// By default calls batched version with size 1.
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs.
    async fn insert_and_trim<V>(
        &mut self,
        store: &mut V,
        to: Self::Vector,
        dist: Self::Distance,
        k: Option<usize>,
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>,
    {
        self.insert_batch_and_trim(store, &[(to, dist)], k).await
    }

    /// Returns matching records in the neighborhood.
    /// No specific order should be assumed.
    async fn matches<V>(&self, store: &mut V) -> Result<Vec<(Self::Vector, Self::Distance)>>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>;

    // Returns a suitable node for opening
    fn get_next_candidate(&self) -> Option<&(Self::Vector, Self::Distance)>;

    /// Returns the node with maximum distance in the neighborhood
    fn get_furthest(&self) -> Option<&(Self::Vector, Self::Distance)>;
}

/// A sorted list of edge IDs (without distances).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct SortedEdgeIds<V>(pub Vec<V>);

impl<V> SortedEdgeIds<V> {
    pub fn from_ascending_vec(edges: Vec<V>) -> Self {
        SortedEdgeIds(edges)
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.0.truncate(k);
    }
}

impl<V> Default for SortedEdgeIds<V> {
    fn default() -> Self {
        SortedEdgeIds(vec![])
    }
}

impl<V> Deref for SortedEdgeIds<V> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> DerefMut for SortedEdgeIds<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub type SortedNeighborhoodV<V> =
    SortedNeighborhood<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;

struct SortedNeighborhoodV2<V: VectorStore>(SortedNeighborhood<V::VectorRef, V::DistanceRef>);

impl<V: VectorStore> NeighborhoodV<V> for SortedNeighborhoodV<V> {}

// impl<Vector: Clone, Distance: Clone> NeighborhoodV2 for SortedNeighborhood<Vector, Distance> {
//     type V = W;
// }

/// SortedNeighborhood maintains a collection of distance-weighted oriented
/// graph edges for an HNSW graph which are stored in increasing order of edge
/// weights. Each edge is stored as a vector id representing the target of the
/// directed edge, along with a cached representation of the distance between
/// the source vector and the target vector. Note that the source vector id is
/// not stored, as this would increase the memory required to store an edge, and
/// is implicit from the context in which the neighborhood is being accessed.
///
/// The aim of ordering operations for this implementation is to prioritize a
/// combination of low total overall comparison operations and low sequential
/// complexity. This reflects the fact that in our usage, these operations
/// will make use of expensive SMPC protocols, so overall comparison counts
/// determine overall network bandwidth usage, and sequential complexity
/// determines serial latency of operations.
#[derive(Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortedNeighborhood<Vector, Distance> {
    /// List of distance-weighted directed edges, specified as tuples
    /// `(target, weight)`. Edges are sorted in increasing order of distance.
    pub edges: Vec<(Vector, Distance)>,
}

impl<Vector: Clone, Distance: Clone> SortedNeighborhood<Vector, Distance> {
    pub fn from_ascending_vec(edges: Vec<(Vector, Distance)>) -> Self {
        SortedNeighborhood { edges }
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(Vector, Distance)] {
        &self.edges[..k]
    }

    pub fn as_vec_ref(&self) -> &[(Vector, Distance)] {
        &self.edges
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using the Batcher odd-even merge sort sorting
    /// network.
    #[allow(dead_code)]
    async fn batcher_insert<V>(&mut self, store: &mut V, vals: &[(Vector, Distance)]) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let sorted_prefix_size = self.edges.len();
        let unsorted_size = vals.len();

        self.edges.extend_from_slice(vals);
        let network = partial_batcher_network(sorted_prefix_size, unsorted_size)?;

        apply_swap_network(store, &mut self.edges, &network).await
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using a parallelized quicksort algorithm.
    async fn quicksort_insert<V>(
        &mut self,
        store: &mut V,
        vals: &[(Vector, Distance)],
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let sorted_prefix_size = self.edges.len();

        self.edges.extend_from_slice(vals);
        let mut buffer = self.edges.clone();

        apply_quicksort(store, &mut self.edges, &mut buffer, sorted_prefix_size).await
    }
}

#[allow(async_fn_in_trait)]
impl<Vector, Distance> Neighborhood for SortedNeighborhood<Vector, Distance>
where
    Vector: Clone,
    Distance: Clone,
{
    type Vector = Vector;
    type Distance = Distance;
    /// Insert the element `to` with distance `dist` into the list, maintaining
    /// the ascending order.
    ///
    /// Calls the `VectorStore` to find the insertion index.
    fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    fn iter(&self) -> impl Iterator<Item = &(Vector, Distance)> {
        self.edges.iter()
    }

    fn clear(&mut self) {
        self.edges.clear();
    }

    /// Appends `vals` to the neighborhood, applies quicksort and truncates to length `k`.
    /// Expected bandwitdh: loglinear in `|self.edges| + |vals|`, but `|vals| log |self.edges|` for small `|vals|`.
    /// Expected rounds: logarithmic in `|self.edges| + |vals|`.
    /// Consult quicksort implementation for details on performance.
    async fn insert_batch_and_trim<V>(
        &mut self,
        store: &mut V,
        vals: &[(Self::Vector, Self::Distance)],
        k: Option<usize>,
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>,
    {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");

        // Note that quicksort insert does not suffer from reduced performance
        // for small batch sizes, as the functionality gracefully degrades to
        // the default individual binary insertion procedure as batch size
        // approaches 1.
        self.quicksort_insert(store, vals).await?;
        let k = k.unwrap_or(self.edges.len());
        self.edges.truncate(k);
        Ok(())
    }

    fn get_next_candidate(&self) -> Option<&(Self::Vector, Self::Distance)> {
        self.edges.first()
    }

    fn get_furthest(&self) -> Option<&(Self::Vector, Self::Distance)> {
        self.edges.last()
    }
    /// Count the neighbors that match according to `store.is_match`.
    /// The nearest `count` elements are matches and the rest are non-matches.
    async fn matches<V>(&self, store: &mut V) -> Result<Vec<(Vector, Distance)>>
    where
        V: VectorStore<VectorRef = Self::Vector, DistanceRef = Self::Distance>,
    {
        let mut left = 0;
        let mut right = self.edges.len();

        while left < right {
            let mid = left + (right - left) / 2;
            let (_, distance) = &self.edges[mid];

            match store.is_match(distance).await? {
                true => left = mid + 1,
                false => right = mid,
            }
        }
        let matches = self.edges.iter().take(left).cloned().collect::<Vec<_>>();
        Ok(matches)
    }
}

impl<Vector, Distance> Deref for SortedNeighborhood<Vector, Distance> {
    type Target = [(Vector, Distance)];

    fn deref(&self) -> &Self::Target {
        &self.edges
    }
}

impl<Vector: Clone, Distance: Clone> Clone for SortedNeighborhood<Vector, Distance> {
    fn clone(&self) -> Self {
        SortedNeighborhood {
            edges: self.edges.clone(),
        }
    }
}

/// An unsorted list of edge IDs (without distances).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct UnsortedEdgeIds<V>(pub Vec<V>);

impl<V> UnsortedEdgeIds<V> {
    pub fn from_vec(edges: Vec<V>) -> Self {
        UnsortedEdgeIds(edges)
    }
}

impl<V> Default for UnsortedEdgeIds<V> {
    fn default() -> Self {
        UnsortedEdgeIds(vec![])
    }
}

impl<V> Deref for UnsortedEdgeIds<V> {
    type Target = Vec<V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> DerefMut for UnsortedEdgeIds<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
pub struct UnsortedNeighborhood<Vector, Distance> {
    /// List of distance-weighted directed edges, specified as tuples
    /// `(target, weight)`
    pub edges: Vec<(Vector, Distance)>,
}

impl<Vector: Clone, Distance: Clone> UnsortedNeighborhood<Vector, Distance> {
    async fn quickselect<V>(&mut self, store: &mut V, k: usize) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        assert!(0 < k && k <= self.edges.len());
        let values = self
            .edges
            .iter()
            .map(|(_, dist)| dist.clone())
            .collect::<Vec<_>>();
        // Get the permutation which describes the partitioned sequence
        let indices = run_quickselect_with_store(store, &values, k).await?;
        self.edges = indices
            .into_iter()
            .take(k) // truncate to smallest k
            .map(|i| self.edges[i].clone())
            .collect::<Vec<_>>();
        Ok(())
    }
}

pub type UnsortedNeighborhoodV<V> =
    UnsortedNeighborhood<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;

impl<V: VectorStore> NeighborhoodV<V> for UnsortedNeighborhoodV<V> {}

impl<Vector: Clone, Distance: Clone> Neighborhood for UnsortedNeighborhood<Vector, Distance> {
    type Vector = Vector;
    type Distance = Distance;
    fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &(Vector, Distance)> {
        self.edges.iter()
    }

    fn len(&self) -> usize {
        self.edges.len()
    }

    fn clear(&mut self) {
        self.edges.clear();
    }

    /// Appends `vals` to the neighborhood, applies quickselect and truncates to length k.
    /// Expected bandwitdh: linear in `|self.edges| + |vals|`
    /// Expected rounds: logarithmic in `|self.edges| + |vals|`
    async fn insert_batch_and_trim<V>(
        &mut self,
        store: &mut V,
        vals: &[(Vector, Distance)],
        k: Option<usize>,
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");
        self.edges.extend(vals.to_vec());

        let k = k.unwrap_or(self.edges.len());
        let k = k.min(self.edges.len());

        if k == 0 {
            self.edges.clear();
        } else {
            self.quickselect(store, k).await?;
        }
        Ok(())
    }

    fn get_next_candidate(&self) -> Option<&(Self::Vector, Self::Distance)> {
        self.edges.first()
    }

    fn get_furthest(&self) -> Option<&(Vector, Distance)> {
        self.edges.last()
    }

    /// Count the neighbors that match according to `store.is_match`.
    async fn matches<V>(&self, store: &mut V) -> Result<Vec<(Vector, Distance)>>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let distances = self
            .edges
            .iter()
            .map(|(_, dist)| dist.clone())
            .collect::<Vec<_>>();
        let results = store.is_match_batch(&distances).await?;
        Ok(self
            .edges
            .iter()
            .enumerate()
            .filter_map(|(i, (v, d))| {
                if results[i] {
                    Some((v.clone(), d.clone()))
                } else {
                    None
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::vector_store::VectorStoreMut};
    use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
    use rand::thread_rng;

    async fn insert_batch_store_and_nbhd<N: NeighborhoodV<PlaintextStore>>(
        query: Arc<IrisCode>,
        store: &mut PlaintextStore,
        nbhd: &mut N,
        irises: &[Arc<IrisCode>],
    ) -> Result<()> {
        let mut pairs = Vec::new();
        for iris in irises {
            let vector = store.insert(iris).await;
            let distance = store.eval_distance(&query, &vector).await?;
            pairs.push((vector, distance));
        }

        // This is a bit wasteful but we don't care
        nbhd.insert_batch_and_trim(store, &pairs, None).await?;
        Ok(())
    }

    async fn test_neighborhood_generic<N: NeighborhoodV<PlaintextStore>>() -> Result<()> {
        let mut rng = thread_rng();
        let mut store = PlaintextStore::new();
        let query = Arc::new(IrisCode::random_rng(&mut rng));
        let mut nbhd = N::new();

        let randos = (0..3)
            .map(|_| Arc::new(IrisCode::random_rng(&mut rng)))
            .collect::<Vec<_>>();
        insert_batch_store_and_nbhd(query.clone(), &mut store, &mut nbhd, &randos).await?;

        assert_eq!(nbhd.len(), 3);
        // Insert a match
        insert_batch_store_and_nbhd(
            query.clone(),
            &mut store,
            &mut nbhd,
            &[Arc::new(query.get_similar_iris(&mut rng, 0.18))],
        )
        .await?;
        nbhd.insert_batch_and_trim(&mut store, &[], Some(2)).await?;
        assert_eq!(nbhd.len(), 2);

        // We should have exactly one match at this point (almost always :))
        assert_eq!(nbhd.matches(&mut store).await?.len(), 1);

        // Next candidate is arbitrary in general, but it should not be None
        _ = nbhd.get_next_candidate().unwrap();

        // Insert an even closer match
        insert_batch_store_and_nbhd(
            query.clone(),
            &mut store,
            &mut nbhd,
            &[Arc::new(query.get_similar_iris(&mut rng, 0.05))],
        )
        .await?;
        assert_eq!(nbhd.len(), 3);

        // Let's shrink to closest 2
        nbhd.insert_batch_and_trim(&mut store, &[], Some(2)).await?;
        assert_eq!(nbhd.len(), 2);

        // We should have exactly two matches at this point
        assert_eq!(nbhd.matches(&mut store).await?.len(), 2);

        // The furthest element should be the earliest inserted match
        let furthest = nbhd.get_furthest().unwrap();
        assert_eq!(furthest.0, IrisVectorId::from_serial_id(4));

        // This should clear
        nbhd.insert_batch_and_trim(&mut store, &[], Some(0)).await?;

        assert_eq!(nbhd.len(), 0);
        assert!(nbhd.is_empty());
        assert!(nbhd.get_furthest().is_none());
        assert!(nbhd.get_next_candidate().is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_sorted_neighborhood() {
        test_neighborhood_generic::<SortedNeighborhoodV<PlaintextStore>>()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_unsorted_neighborhood() {
        test_neighborhood_generic::<UnsortedNeighborhoodV<PlaintextStore>>()
            .await
            .unwrap();
    }
}
