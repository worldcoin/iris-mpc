//! Implementation of sorted graph neighborhoods for an HNSW hierarchical graph;
//! based on the `FurthestQueue` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::hnsw::{
    sorting::{batcher::partial_batcher_network, sorting_network::SortingNetwork},
    VectorStore,
};
use serde::{Deserialize, Serialize};
use std::ops::Deref;
use tracing::instrument;

pub type SortedNeighborhoodV<V> =
    SortedNeighborhood<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;

/// SortedNeighborhood maintains a collection of distance-weighted oriented
/// graph edges for an HNSW graph which are stored in increasing order of edge
/// weights.  Each edge is stored as a vector id representing the target of the
/// directed edge, along with a cached representation of the distance between
/// the source vector and the target vector.  Note that the source vector id is
/// not stored, as this would increase the memory required to store an edge, and
/// is implicit from the context in which the neighborhood is being accessed.
///
/// The aim of ordering operations for this implementation is to prioritize a
/// combination of low total overall comparison operations and low sequential
/// complexity.  This reflects the fact that in our usage, these operations
/// will make use of expensive SMPC protocols, so overall comparison counts
/// determine overall network bandwidth usage, and sequential complexity
/// determines serial latency of operations.
#[derive(Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortedNeighborhood<Vector, Distance> {
    pub edges: Vec<(Vector, Distance)>,
}

impl<Vector: Clone, Distance: Clone> SortedNeighborhood<Vector, Distance> {
    pub fn new() -> Self {
        Self {
            edges: Default::default(),
        }
    }

    pub fn from_ascending_vec(edges: Vec<(Vector, Distance)>) -> Self {
        SortedNeighborhood { edges }
    }

    /// Insert the element `to` with distance `dist` into the list, maintaining
    /// the ascending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn insert<V>(&mut self, store: &mut V, to: Vector, dist: Distance)
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let index_asc = Self::binary_search(
            store,
            &self
                .edges
                .iter()
                .map(|(_, dist)| dist.clone())
                .collect::<Vec<Distance>>(),
            &dist,
        )
        .await;
        self.edges.insert(index_asc, (to, dist));
    }

    /// Insert a collection of (Vector, Distance) pairs into the list,
    /// maintaining the ascending order, using an efficient sorting network on
    /// input values.
    ///
    /// TODO: give heuristic for when batched insertion is more efficient than
    /// iterated single insertion
    pub async fn insert_batch<V>(&mut self, store: &mut V, vals: &[(Vector, Distance)])
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let sorted_prefix_size = self.edges.len();
        let unsorted_size = vals.len();

        self.edges.extend_from_slice(vals);
        let sorting_network = partial_batcher_network(sorted_prefix_size, unsorted_size);

        self.apply_sorting_network(store, &sorting_network).await;
    }

    pub fn get_nearest(&self) -> Option<&(Vector, Distance)> {
        self.edges.first()
    }

    pub fn get_furthest(&self) -> Option<&(Vector, Distance)> {
        self.edges.last()
    }

    pub fn pop_furthest(&mut self) -> Option<(Vector, Distance)> {
        self.edges.pop()
    }

    pub fn get_k_nearest(&self, k: usize) -> &[(Vector, Distance)] {
        &self.edges[..k]
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.edges.truncate(k);
    }

    pub fn as_vec_ref(&self) -> &[(Vector, Distance)] {
        &self.edges
    }

    async fn apply_sorting_network<V>(&mut self, store: &mut V, network: &SortingNetwork)
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        for layer in network.layers.iter() {
            let distances: Vec<_> = layer
                .iter()
                .filter_map(|(idx1, idx2): &(usize, usize)| {
                    match (self.edges.get(*idx1), self.edges.get(*idx2)) {
                        (Some((_, d1)), Some((_, d2))) => Some((d1.clone(), d2.clone())),
                        _ => None,
                    }
                })
                .collect();
            let comp_results = store.less_than_batch(&distances).await;
            for ((idx1, idx2), is_lt) in layer.iter().zip(comp_results) {
                if is_lt {
                    self.edges.swap(*idx1, *idx2)
                }
            }
        }
    }

    /// Find the insertion index for a target distance in the current
    /// neighborhood list.
    async fn binary_search<V>(store: &mut V, distances: &[Distance], target: &Distance) -> usize
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let mut left = 0;
        let mut right = distances.len();

        while left < right {
            let mid = left + (right - left) / 2;

            match store.less_than(&distances[mid], target).await {
                true => left = mid + 1,
                false => right = mid,
            }
        }
        left
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

impl<Vector, Distance> From<SortedNeighborhood<Vector, Distance>> for Vec<(Vector, Distance)> {
    fn from(nbhd: SortedNeighborhood<Vector, Distance>) -> Self {
        nbhd.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkers::plaintext_store::PlaintextStore;
    use iris_mpc_common::iris_db::iris::IrisCode;

    #[tokio::test]
    async fn test_neighborhood() {
        let mut store = PlaintextStore::default();
        let query = store.prepare_query(IrisCode::default());
        let vector = store.insert(&query).await;
        let distance = store.eval_distance(&query, &vector).await;

        // Example usage for SortedNeighborhood
        let mut nbhd = SortedNeighborhood::new();
        nbhd.insert(&mut store, vector, distance).await;
        println!("{:?}", nbhd.get_furthest());
        println!("{:?}", nbhd.get_k_nearest(1));
        println!("{:?}", nbhd.pop_furthest());
    }
}
