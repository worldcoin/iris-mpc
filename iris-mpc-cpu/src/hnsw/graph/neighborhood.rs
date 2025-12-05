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
use delegate::delegate;
use eyre::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

/// Trait that captures the requirements for an HNSW candidate list container/data-structure.
/// Implementers should ensure the following post-condition for calls to `trim` methods:
/// - The elements in the container are the smallest `container.length` ones among all insertions.
/// - The last element in the container is the largest one.
#[allow(async_fn_in_trait)]
pub trait Neighborhood<V: VectorStore>:
    Send + Sync + Clone + AsRef<[(V::VectorRef, V::DistanceRef)]> + Into<WrappedNeighborhood<V>>
{
    fn new() -> Self;

    fn len(&self) -> usize {
        self.as_ref().len()
    }

    fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    fn from_singleton(element: (V::VectorRef, V::DistanceRef)) -> Self;

    fn edge_ids(&self) -> Vec<V::VectorRef> {
        self.as_ref()
            .iter()
            .map(|(v, _)| v.clone())
            .collect::<Vec<_>>()
    }

    /// Inserts a batch of elements into the neighborhood and applies the necessary
    /// changes to ensure the neighborhood invariant holds
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs
    async fn insert_batch_and_trim(
        &mut self,
        store: &mut V,
        vals: &[(V::VectorRef, V::DistanceRef)],
        k: usize,
    ) -> Result<()>;

    /// Apply the invariant-maintaining algorithm
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs.
    async fn trim(&mut self, store: &mut V, k: usize) -> Result<()> {
        self.insert_batch_and_trim(store, &[], k).await
    }

    /// Inserts a single element into the neighborhood.
    /// By default calls batched version with size 1.
    /// Note that in general maintaining the invariant may incur significant overhead.
    /// Consult implementer for performance specs.
    async fn insert_and_trim(
        &mut self,
        store: &mut V,
        to: V::VectorRef,
        dist: V::DistanceRef,
        k: usize,
    ) -> Result<()> {
        self.insert_batch_and_trim(store, &[(to, dist)], k).await
    }

    /// Returns matching records in the neighborhood.
    /// No specific order should be assumed.
    async fn matches(&self, store: &mut V) -> Result<Vec<(V::VectorRef, V::DistanceRef)>>;

    /// Returns the node with maximum distance in the neighborhood
    fn get_furthest(&self) -> Option<&(V::VectorRef, V::DistanceRef)>;
}

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
pub struct SortedNeighborhood<V: VectorStore> {
    /// List of distance-weighted directed edges, specified as tuples
    /// `(target, weight)`. Edges are sorted in increasing order of distance.
    pub edges: Vec<(V::VectorRef, V::DistanceRef)>,
}

impl<V: VectorStore> Clone for SortedNeighborhood<V>
where
    V::VectorRef: Clone,
    V::DistanceRef: Clone,
{
    fn clone(&self) -> Self {
        SortedNeighborhood {
            edges: self.edges.clone(),
        }
    }
}

impl<V: VectorStore> SortedNeighborhood<V> {
    pub fn from_ascending_vec(edges: Vec<(V::VectorRef, V::DistanceRef)>) -> Self {
        SortedNeighborhood { edges }
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(V::VectorRef, V::DistanceRef)] {
        &self.edges[..k]
    }

    pub fn as_vec_ref(&self) -> &[(V::VectorRef, V::DistanceRef)] {
        &self.edges
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using the Batcher odd-even merge sort sorting
    /// network.
    #[allow(dead_code)]
    async fn batcher_insert(
        &mut self,
        store: &mut V,
        vals: &[(V::VectorRef, V::DistanceRef)],
    ) -> Result<()> {
        let sorted_prefix_size = self.edges.len();
        let unsorted_size = vals.len();

        self.edges.extend_from_slice(vals);
        let network = partial_batcher_network(sorted_prefix_size, unsorted_size)?;

        apply_swap_network(store, &mut self.edges, &network).await
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using a parallelized quicksort algorithm.
    async fn quicksort_insert(
        &mut self,
        store: &mut V,
        vals: &[(V::VectorRef, V::DistanceRef)],
    ) -> Result<()> {
        let sorted_prefix_size = self.edges.len();

        self.edges.extend_from_slice(vals);
        let mut buffer = self.edges.clone();

        apply_quicksort(store, &mut self.edges, &mut buffer, sorted_prefix_size).await
    }
}

impl<V: VectorStore> AsRef<[(V::VectorRef, V::DistanceRef)]> for SortedNeighborhood<V> {
    fn as_ref(&self) -> &[(V::VectorRef, V::DistanceRef)] {
        &self.edges
    }
}

#[allow(async_fn_in_trait)]
impl<V: VectorStore> Neighborhood<V> for SortedNeighborhood<V> {
    /// Insert the element `to` with distance `dist` into the list, maintaining
    /// the ascending order.
    ///
    /// Calls the `VectorStore` to find the insertion index.
    fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    fn from_singleton(element: (<V as VectorStore>::VectorRef, V::DistanceRef)) -> Self {
        SortedNeighborhood {
            edges: vec![element],
        }
    }

    /// Appends `vals` to the neighborhood, applies quicksort and truncates to length `k`.
    /// Expected bandwitdh: loglinear in `|self.edges| + |vals|`, but `|vals| log |self.edges|` for small `|vals|`.
    /// Expected rounds: logarithmic in `|self.edges| + |vals|`.
    /// Consult quicksort implementation for details on performance.
    async fn insert_batch_and_trim(
        &mut self,
        store: &mut V,
        vals: &[(V::VectorRef, V::DistanceRef)],
        k: usize,
    ) -> Result<()> {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");

        // Note that quicksort insert does not suffer from reduced performance
        // for small batch sizes, as the functionality gracefully degrades to
        // the default individual binary insertion procedure as batch size
        // approaches 1.
        self.quicksort_insert(store, vals).await?;
        self.edges.truncate(k);
        Ok(())
    }

    fn get_furthest(&self) -> Option<&(V::VectorRef, V::DistanceRef)> {
        self.edges.last()
    }
    /// Count the neighbors that match according to `store.is_match`.
    /// The nearest `count` elements are matches and the rest are non-matches.
    async fn matches(&self, store: &mut V) -> Result<Vec<(V::VectorRef, V::DistanceRef)>> {
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

pub struct UnsortedNeighborhood<V: VectorStore> {
    /// List of distance-weighted directed edges, specified as tuples
    /// `(target, weight)`
    pub edges: Vec<(V::VectorRef, V::DistanceRef)>,
}

impl<V: VectorStore> Clone for UnsortedNeighborhood<V>
where
    V::VectorRef: Clone,
    V::DistanceRef: Clone,
{
    fn clone(&self) -> Self {
        UnsortedNeighborhood {
            edges: self.edges.clone(),
        }
    }
}

impl<V: VectorStore> UnsortedNeighborhood<V> {
    async fn quickselect(&mut self, store: &mut V, k: usize) -> Result<()> {
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

impl<V: VectorStore> AsRef<[(V::VectorRef, V::DistanceRef)]> for UnsortedNeighborhood<V> {
    fn as_ref(&self) -> &[(V::VectorRef, V::DistanceRef)] {
        &self.edges
    }
}

impl<V: VectorStore> Neighborhood<V> for UnsortedNeighborhood<V> {
    fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    fn from_singleton(element: (<V as VectorStore>::VectorRef, V::DistanceRef)) -> Self {
        UnsortedNeighborhood {
            edges: vec![element],
        }
    }

    /// Appends `vals` to the neighborhood, applies quickselect and truncates to length k.
    /// Expected bandwitdh: linear in `|self.edges| + |vals|`
    /// Expected rounds: logarithmic in `|self.edges| + |vals|`
    async fn insert_batch_and_trim(
        &mut self,
        store: &mut V,
        vals: &[(V::VectorRef, V::DistanceRef)],
        k: usize,
    ) -> Result<()> {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");
        self.edges.extend(vals.to_vec());

        // TODO: this case can be optimized if needed
        let k = k.min(self.edges.len());

        if k == 0 {
            self.edges.clear();
        } else {
            self.quickselect(store, k).await?;
        }
        Ok(())
    }

    fn get_furthest(&self) -> Option<&(V::VectorRef, V::DistanceRef)> {
        self.edges.last()
    }

    /// Count the neighbors that match according to `store.is_match`.
    async fn matches(&self, store: &mut V) -> Result<Vec<(V::VectorRef, V::DistanceRef)>> {
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

/// Datatype which covers all used implementers of Neighborhood
#[allow(dead_code)]
pub enum WrappedNeighborhood<V: VectorStore> {
    Sorted(SortedNeighborhood<V>),
    Unsorted(UnsortedNeighborhood<V>),
}

impl<V: VectorStore> From<SortedNeighborhood<V>> for WrappedNeighborhood<V> {
    fn from(value: SortedNeighborhood<V>) -> Self {
        Self::Sorted(value)
    }
}

impl<V: VectorStore> From<UnsortedNeighborhood<V>> for WrappedNeighborhood<V> {
    fn from(value: UnsortedNeighborhood<V>) -> Self {
        Self::Unsorted(value)
    }
}

#[allow(dead_code)]
impl<V: VectorStore> WrappedNeighborhood<V> {
    delegate! {
        to match self {
            WrappedNeighborhood::Sorted(nb) => nb,
            WrappedNeighborhood::Unsorted(nb) => nb,
        } {
            async fn insert_batch_and_trim(
                &mut self,
                store: &mut V,
                vals: &[(V::VectorRef, V::DistanceRef)],
                k: usize,
            ) -> Result<()>;

            async fn matches(&self, store: &mut V) -> Result<Vec<(V::VectorRef, V::DistanceRef)>>;

            fn get_furthest(&self) -> Option<&(V::VectorRef, V::DistanceRef)>;
        }
    }
}

impl<V: VectorStore> AsRef<[(V::VectorRef, V::DistanceRef)]> for WrappedNeighborhood<V> {
    fn as_ref(&self) -> &[(V::VectorRef, V::DistanceRef)] {
        match self {
            WrappedNeighborhood::Sorted(nb) => nb.as_ref(),
            WrappedNeighborhood::Unsorted(nb) => nb.as_ref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::vector_store::VectorStoreMut};
    use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
    use rand::thread_rng;

    async fn insert_batch_store_and_nbhd<N: Neighborhood<PlaintextStore>>(
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
        nbhd.insert_batch_and_trim(store, &pairs, nbhd.len() + pairs.len())
            .await?;
        Ok(())
    }

    async fn test_neighborhood_generic<N: Neighborhood<PlaintextStore>>() -> Result<()> {
        let mut rng = thread_rng();
        let mut store = PlaintextStore::new();
        let query = Arc::new(IrisCode::random_rng(&mut rng));
        let mut nbhd = N::new();

        let randos = (0..3)
            .map(|_| Arc::new(IrisCode::random_rng(&mut rng)))
            .collect::<Vec<_>>();
        insert_batch_store_and_nbhd(query.clone(), &mut store, &mut nbhd, &randos).await?;

        assert_eq!(nbhd.as_ref().len(), 3);
        // Insert a match
        insert_batch_store_and_nbhd(
            query.clone(),
            &mut store,
            &mut nbhd,
            &[Arc::new(query.get_similar_iris(&mut rng, 0.18))],
        )
        .await?;
        nbhd.insert_batch_and_trim(&mut store, &[], 2).await?;
        assert_eq!(nbhd.as_ref().len(), 2);

        // We should have exactly one match at this point (almost always :))
        assert_eq!(nbhd.matches(&mut store).await?.len(), 1);

        // Next candidate is arbitrary in general, but it should not be None

        // Insert an even closer match
        insert_batch_store_and_nbhd(
            query.clone(),
            &mut store,
            &mut nbhd,
            &[Arc::new(query.get_similar_iris(&mut rng, 0.05))],
        )
        .await?;
        assert_eq!(nbhd.as_ref().len(), 3);

        // Let's shrink to closest 2
        nbhd.insert_batch_and_trim(&mut store, &[], 2).await?;
        assert_eq!(nbhd.as_ref().len(), 2);

        // We should have exactly two matches at this point
        assert_eq!(nbhd.matches(&mut store).await?.len(), 2);

        // The furthest element should be the earliest inserted match
        let furthest = nbhd.get_furthest().unwrap();
        assert_eq!(furthest.0, IrisVectorId::from_serial_id(4));

        // This should clear
        nbhd.insert_batch_and_trim(&mut store, &[], 0).await?;

        assert_eq!(nbhd.as_ref().len(), 0);
        assert!(nbhd.as_ref().is_empty());
        assert!(nbhd.get_furthest().is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_sorted_neighborhood() {
        test_neighborhood_generic::<SortedNeighborhood<PlaintextStore>>()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_unsorted_neighborhood() {
        test_neighborhood_generic::<UnsortedNeighborhood<PlaintextStore>>()
            .await
            .unwrap();
    }
}
