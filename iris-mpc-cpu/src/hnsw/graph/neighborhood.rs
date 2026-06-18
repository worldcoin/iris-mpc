//! Implementation of sorted graph neighborhoods for an HNSW hierarchical graph;
//! based on the `FurthestQueue` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::hnsw::{
    sorting::{
        batcher::partial_batcher_network, quicksort::apply_quicksort,
        swap_network::apply_swap_network,
    },
    VectorStore,
};
use eyre::Result;
use iris_mpc_common::VectorId;
use serde::{Deserialize, Serialize};
use tracing::debug;

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
    pub edges: Vec<(VectorId, V::DistanceRef)>,
}

impl<V: VectorStore> Clone for SortedNeighborhood<V>
where
    VectorId: Clone,
    V::DistanceRef: Clone,
{
    fn clone(&self) -> Self {
        SortedNeighborhood {
            edges: self.edges.clone(),
        }
    }
}

impl<V: VectorStore> SortedNeighborhood<V> {
    pub fn from_ascending_vec(edges: Vec<(VectorId, V::DistanceRef)>) -> Self {
        SortedNeighborhood { edges }
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(VectorId, V::DistanceRef)] {
        &self.edges[..k]
    }

    pub fn as_vec_ref(&self) -> &[(VectorId, V::DistanceRef)] {
        &self.edges
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using the Batcher odd-even merge sort sorting
    /// network.
    #[allow(dead_code)]
    async fn batcher_insert(
        &mut self,
        store: &mut V,
        vals: &[(VectorId, V::DistanceRef)],
    ) -> Result<()> {
        let sorted_prefix_size = self.edges.len();
        let unsorted_size = vals.len();

        self.edges.extend_from_slice(vals);
        let network = partial_batcher_network(sorted_prefix_size, unsorted_size)?;

        apply_swap_network(store, &mut self.edges, &network).await
    }

    /// Insert the given unsorted list `vals` of new weighted edges into this
    /// sorted neighborhood using a parallelized quicksort algorithm.
    ///
    /// When `truncate_k` is `Some(k)`, the underlying quicksort skips
    /// recursive subsorts that fall entirely past the truncation index, so
    /// only the first `k` elements end up in fully sorted order.
    async fn quicksort_insert(
        &mut self,
        store: &mut V,
        vals: &[(VectorId, V::DistanceRef)],
        truncate_k: Option<usize>,
    ) -> Result<()> {
        let sorted_prefix_size = self.edges.len();

        self.edges.extend_from_slice(vals);
        let mut buffer = self.edges.clone();

        apply_quicksort(
            store,
            &mut self.edges,
            &mut buffer,
            sorted_prefix_size,
            truncate_k,
        )
        .await
    }
}

impl<V: VectorStore> AsRef<[(VectorId, V::DistanceRef)]> for SortedNeighborhood<V> {
    fn as_ref(&self) -> &[(VectorId, V::DistanceRef)] {
        &self.edges
    }
}

impl<V: VectorStore> SortedNeighborhood<V> {
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    pub fn len(&self) -> usize {
        self.edges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn from_singleton(element: (VectorId, V::DistanceRef)) -> Self {
        SortedNeighborhood {
            edges: vec![element],
        }
    }

    pub fn edge_ids(&self) -> Vec<VectorId> {
        self.edges.iter().map(|(v, _)| *v).collect::<Vec<_>>()
    }

    /// Apply the invariant-maintaining algorithm so the elements are the
    /// smallest `k` ones among all insertions, with the last element largest.
    pub async fn trim(&mut self, store: &mut V, k: usize) -> Result<()> {
        self.insert_batch_and_trim(store, &[], k).await
    }

    /// Inserts a single element into the neighborhood and trims to length `k`.
    pub async fn insert_and_trim(
        &mut self,
        store: &mut V,
        to: VectorId,
        dist: V::DistanceRef,
        k: usize,
    ) -> Result<()> {
        self.insert_batch_and_trim(store, &[(to, dist)], k).await
    }

    /// Appends `vals` to the neighborhood, applies quicksort and truncates to length `k`.
    /// Expected bandwidth: loglinear in `|self.edges| + |vals|`, but `|vals| log |self.edges|` for small `|vals|`.
    /// Expected rounds: logarithmic in `|self.edges| + |vals|`.
    /// Consult quicksort implementation for details on performance.
    pub async fn insert_batch_and_trim(
        &mut self,
        store: &mut V,
        vals: &[(VectorId, V::DistanceRef)],
        k: usize,
    ) -> Result<()> {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");

        // Note that quicksort insert does not suffer from reduced performance
        // for small batch sizes, as the functionality gracefully degrades to
        // the default individual binary insertion procedure as batch size
        // approaches 1.
        //
        // We pass `Some(k)` so the quicksort skips sorting work in the suffix
        // that the subsequent `truncate(k)` would discard.
        self.quicksort_insert(store, vals, Some(k)).await?;
        self.edges.truncate(k);
        Ok(())
    }

    pub fn get_furthest(&self) -> Option<&(VectorId, V::DistanceRef)> {
        self.edges.last()
    }
    /// Count the neighbors that match according to `store.is_match`.
    /// The nearest `count` elements are matches and the rest are non-matches.
    pub async fn matches(&self, store: &mut V) -> Result<Vec<(VectorId, V::DistanceRef)>> {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::vector_store::VectorStoreMut};
    use iris_mpc_common::{iris_db::iris::IrisCode, VectorId};
    use rand::thread_rng;

    async fn insert_batch_store_and_nbhd(
        query: Arc<IrisCode>,
        store: &mut PlaintextStore,
        nbhd: &mut SortedNeighborhood<PlaintextStore>,
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

    #[tokio::test]
    async fn test_sorted_neighborhood() -> Result<()> {
        let mut rng = thread_rng();
        let mut store = PlaintextStore::new();
        let query = Arc::new(IrisCode::random_rng(&mut rng));
        let mut nbhd = SortedNeighborhood::<PlaintextStore>::new();

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
        assert_eq!(furthest.0, VectorId::from_serial_id(4));

        // This should clear
        nbhd.insert_batch_and_trim(&mut store, &[], 0).await?;

        assert_eq!(nbhd.as_ref().len(), 0);
        assert!(nbhd.as_ref().is_empty());
        assert!(nbhd.get_furthest().is_none());

        Ok(())
    }
}
