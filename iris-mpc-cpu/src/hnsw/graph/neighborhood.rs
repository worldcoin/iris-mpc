//! Implementation of sorted graph neighborhoods for an HNSW hierarchical graph;
//! based on the `FurthestQueue` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::hnsw::{
    searcher::{
        ConnectPlan, ConnectPlanLayer, ConnectPlanV, LightConnectPlanLayer, LightConnectPlanV,
    },
    sorting::{
        batcher::partial_batcher_network, binary_search::BinarySearch, quicksort::apply_quicksort,
        swap_network::apply_swap_network,
    },
    vector_store::Ref,
    GraphMem, HnswSearcher, VectorStore,
};
use eyre::{eyre, Result};
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
    str::FromStr,
};
use tracing::{debug, instrument};

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
    /// Calls the `VectorStore` to find the insertion index.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn insert<V>(&mut self, store: &mut V, to: Vector, dist: Distance) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let mut bin_search = BinarySearch {
            left: 0,
            right: self.edges.len(),
        };
        while let Some(cmp_idx) = bin_search.next() {
            let res = store.less_than(&dist, &self.edges[cmp_idx].1).await?;
            bin_search.update(res);
        }
        let index_asc = bin_search
            .result()
            .ok_or(eyre!("Failed to find insertion index"))?;
        self.edges.insert(index_asc, (to, dist));
        Ok(())
    }

    /// Insert a collection of `(Vector, Distance)` pairs into the list,
    /// maintaining the ascending order, using an efficient sorting network on
    /// input values.
    pub async fn insert_batch<V>(
        &mut self,
        store: &mut V,
        vals: &[(Vector, Distance)],
    ) -> Result<()>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        debug!(batch_size = vals.len(), "Insert batch into neighborhood");

        if vals.is_empty() {
            return Ok(());
        }

        // Note that quicksort insert does not suffer from reduced performance
        // for small batch sizes, as the functionality gracefully degrades to
        // the default individual binary insertion procedure as batch size
        // approaches 1.
        self.quicksort_insert(store, vals).await
    }

    pub fn edge_ids(&self) -> SortedEdgeIds<Vector> {
        SortedEdgeIds(self.vectors_cloned())
    }

    pub fn vectors_cloned(&self) -> Vec<Vector> {
        self.edges.iter().map(|(v, _)| v.clone()).collect()
    }

    pub fn distances_cloned(&self) -> Vec<Distance> {
        self.edges.iter().map(|(_, d)| d.clone()).collect()
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

    /// Count the neighbors that match according to `store.is_match`.
    /// The nearest `count` elements are matches and the rest are non-matches.
    pub async fn match_count<V>(&self, store: &mut V) -> Result<usize>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
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

        Ok(left)
    }
}

impl<Vector: Ref + Display + FromStr, Distance: Clone> SortedNeighborhood<Vector, Distance> {
    pub async fn batch_insert_prepare<V>(
        instances: Vec<(Vector, Vec<SortedNeighborhoodV<V>>, bool)>,
        store: &mut V,
        graph: &GraphMem<Vector>,
    ) -> Result<Vec<LightConnectPlanV<V>>>
    where
        V: VectorStore<VectorRef = Vector, DistanceRef = Distance>,
    {
        let mut plans = instances
            .iter()
            .map(|(inserted_vector, _, set_ep)| ConnectPlan {
                inserted_vector: inserted_vector.clone(),
                layers: vec![],
                set_ep: *set_ep,
            })
            .collect::<Vec<_>>();

        struct NeighborUpdate<Query, Vector, Distance> {
            /// The distance between the vector being inserted to a base vector.
            nb_dist: Distance,
            /// The base vector that we connect to. It is in "query" form to compare to `nb_links`.
            nb_query: Query,
            /// The neighborhood of the base vector.
            nb_links: SortedEdgeIds<Vector>,
            /// The current state of the search.
            search: BinarySearch,
        }

        // Collect current neighborhoods of new neighbors in each layer and
        // initialize binary search
        let mut neighbors = Vec::new();
        for (_, links, _) in instances.iter() {
            for (lc, l_links) in links.iter().enumerate() {
                let nb_queries = store.vectors_as_queries(l_links.vectors_cloned()).await;

                let mut l_neighbors = Vec::with_capacity(l_links.len());
                for ((nb, nb_dist), nb_query) in izip!(l_links.iter(), nb_queries) {
                    let nb_links = graph.get_links(nb, lc).await;
                    let nb_links = SortedEdgeIds(store.only_valid_vectors(nb_links.0).await);
                    let search = BinarySearch {
                        left: 0,
                        right: nb_links.len(),
                    };
                    let neighbor = NeighborUpdate {
                        nb_dist: nb_dist.clone(),
                        nb_query,
                        nb_links,
                        search,
                    };
                    l_neighbors.push(neighbor);
                }
                neighbors.push(l_neighbors);
            }
        }

        // Run searches until completion, executing comparisons in batches
        let mut searches_ongoing: Vec<_> = neighbors
            .iter_mut()
            .flatten()
            .filter(|n| !n.search.is_finished())
            .collect();

        while !searches_ongoing.is_empty() {
            // Find the next batch of distances to evaluate.
            // This is each base neighbor versus the next search position in its neighborhood.
            let dist_batch = searches_ongoing
                .iter()
                .map(|n| {
                    let cmp_idx = n.search.next().ok_or(eyre!("No next index found"))?;
                    Ok((n.nb_query.clone(), n.nb_links[cmp_idx].clone()))
                })
                .collect::<Result<Vec<_>>>()?;

            // Compute the distances.
            let link_distances = store.eval_distance_pairs(&dist_batch).await?;

            // Prepare a batch of less_than.
            // This is |inserted--base| versus |base--neighborhood|.
            let lt_batch = izip!(&searches_ongoing, link_distances)
                .map(|(n, link_dist)| (n.nb_dist.clone(), link_dist))
                .collect::<Vec<_>>();

            // Compute the less_than.
            let results = store.less_than_batch(&lt_batch).await?;

            searches_ongoing
                .iter_mut()
                .zip(results)
                .for_each(|(n, res)| {
                    n.search.update(res);
                });

            searches_ongoing.retain(|n| !n.search.is_finished());
        }

        for ((_, neighbs, _), plan) in izip!(instances, plans.iter_mut()) {
            let v_neighbors = neighbors.drain(0..neighbs.len()).collect::<Vec<_>>();
            // Directly insert new vector into neighborhoods from search results
            let indices = v_neighbors
                .iter()
                .map(|l_neighbors| {
                    l_neighbors
                        .iter()
                        .map(|n| n.search.result().ok_or(eyre!("No insertion index found")))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            // Generate ConnectPlanLayer structs
            plan.layers = neighbs
                .into_iter()
                .zip(indices)
                .map(|(l_links, nb_insert_index)| LightConnectPlanLayer {
                    neighbors: l_links.edge_ids(),
                    nb_insert_index,
                })
                .collect();
        }

        Ok(plans)
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::vector_store::VectorStoreMut};
    use iris_mpc_common::iris_db::iris::IrisCode;

    #[tokio::test]
    async fn test_neighborhood() -> Result<()> {
        let mut store = PlaintextStore::new();
        let query = Arc::new(IrisCode::default());
        let vector = store.insert(&query).await;
        let distance = store.eval_distance(&query, &vector).await?;

        // Example usage for SortedNeighborhood
        let mut nbhd = SortedNeighborhood::new();
        nbhd.insert(&mut store, vector, distance).await?;
        println!("{:?}", nbhd.get_furthest());
        println!("{:?}", nbhd.get_k_nearest(1));
        println!("{:?}", nbhd.pop_furthest());

        Ok(())
    }
}
