//! Implementation of sorted graph neighborhoods for an HNSW hierarchical graph;
//! based on the `FurthestQueue` class of the `hawk-pack` crate:
//!
//! https://github.com/Inversed-Tech/hawk-pack/

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use crate::hawkers::vector_store::VectorStore;

pub type Edge<V> = (
    <V as VectorStore>::VectorRef,
    <V as VectorStore>::DistanceRef,
);

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
pub struct SortedNeighborhood<V: VectorStore> {
    pub queue: Vec<Edge<V>>,
}

impl<V: VectorStore> SortedNeighborhood<V> {
    pub fn new() -> Self {
        Self {
            queue: Default::default(),
        }
    }

    pub fn from_ascending_vec(queue: Vec<Edge<V>>) -> Self {
        SortedNeighborhood { queue }
    }

    /// Insert the element `to` with distance `dist` into the queue, maintaining
    /// the ascending order.
    ///
    /// Call the VectorStore to come up with the insertion index.
    pub async fn insert(&mut self, store: &mut V, to: V::VectorRef, dist: V::DistanceRef) {
        let index_asc = Self::binary_search(
            store,
            &self
                .queue
                .iter()
                .map(|(_, dist)| dist.clone())
                .collect::<Vec<V::DistanceRef>>(),
            &dist,
        )
        .await;
        self.queue.insert(index_asc, (to, dist));
    }

    pub fn get_nearest(&self) -> Option<&Edge<V>> {
        self.queue.first()
    }

    pub fn get_furthest(&self) -> Option<&Edge<V>> {
        self.queue.last()
    }

    pub fn pop_furthest(&mut self) -> Option<Edge<V>> {
        self.queue.pop()
    }

    pub fn get_k_nearest(&self, k: usize) -> &[Edge<V>] {
        &self.queue[..k]
    }

    pub fn trim_to_k_nearest(&mut self, k: usize) {
        self.queue.truncate(k);
    }

    // Assumes that distance map doesn't change the distance metric
    pub fn map<W, F1, F2>(self, vector_map: F1, distance_map: F2) -> SortedNeighborhood<W>
    where
        W: VectorStore,
        F1: Fn(V::VectorRef) -> W::VectorRef,
        F2: Fn(V::DistanceRef) -> W::DistanceRef,
    {
        let queue: Vec<(W::VectorRef, W::DistanceRef)> = self
            .queue
            .iter()
            .cloned()
            .map(|(v, d)| (vector_map(v), distance_map(d)))
            .collect();
        SortedNeighborhood::from_ascending_vec(queue)
    }

    pub fn as_vec_ref(&self) -> &[Edge<V>] {
        &self.queue
    }

    /// Find the insertion index for a target distance in the current
    /// neighborhood list.
    async fn binary_search(
        store: &mut V,
        distances: &[V::DistanceRef],
        target: &V::DistanceRef,
    ) -> usize
    where
        V: VectorStore,
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

impl<V: VectorStore> Deref for SortedNeighborhood<V> {
    type Target = [Edge<V>];

    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

impl<V: VectorStore> Clone for SortedNeighborhood<V> {
    fn clone(&self) -> Self {
        SortedNeighborhood {
            queue: self.queue.clone(),
        }
    }
}

impl<V: VectorStore> From<SortedNeighborhood<V>> for Vec<Edge<V>> {
    fn from(queue: SortedNeighborhood<V>) -> Self {
        queue.queue
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

        // Example usage for FurthestQueue
        let mut furthest_queue = SortedNeighborhood::new();
        furthest_queue.insert(&mut store, vector, distance).await;
        println!("{:?}", furthest_queue.get_furthest());
        println!("{:?}", furthest_queue.get_k_nearest(1));
        println!("{:?}", furthest_queue.pop_furthest());
    }
}
