use hawk_pack::{
    data_structures::queue::{FurthestQueue, FurthestQueueV},
    VectorStore,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Utility struct, not exposed externally because it doesn't have enough use
/// semantically to add an extra abstraction to the public interface.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct EntryPoint<VectorRef> {
    pub point: VectorRef,
    pub layer: usize,
}

#[derive(Default, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct GraphMem<V: VectorStore> {
    entry_point: Option<EntryPoint<V::VectorRef>>,
    layers: Vec<Layer<V>>,
}

impl<V: VectorStore> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_point: None,
            layers: vec![],
        }
    }

    pub fn from_precomputed(
        entry_point: Option<(V::VectorRef, usize)>,
        layers: Vec<Layer<V>>,
    ) -> Self {
        GraphMem {
            entry_point: entry_point.map(|ep| EntryPoint {
                point: ep.0,
                layer: ep.1,
            }),
            layers,
        }
    }

    pub fn get_layers(&self) -> Vec<Layer<V>> {
        self.layers.clone()
    }
}

// Plain converter for a Graph structure that has different distance ref and vector ref types.
// WARNING: distance metric is assumed to stay the same; thus, conversion doesn't change the graph structure.
// Needed when switching from a PlaintextStore to a secret shared VectorStore.
impl<V: VectorStore> GraphMem<V> {
    pub fn from_another<U, F1, F2>(graph: GraphMem<U>, vector_map: F1, distance_map: F2) -> Self
    where
        U: VectorStore,
        F1: Fn(U::VectorRef) -> V::VectorRef + Copy,
        F2: Fn(U::DistanceRef) -> V::DistanceRef + Copy,
    {
        let new_entry_point = graph.entry_point.map(|ep| EntryPoint {
            point: vector_map(ep.point),
            layer: ep.layer,
        });

        let new_layers: Vec<_> = graph
            .layers
            .into_iter()
            .map(|v| {
                let links: HashMap<_, _> = v
                    .links
                    .into_iter()
                    .map(|(v, q)| (vector_map(v), q.map::<V, F1, F2>(vector_map, distance_map)))
                    .collect();
                Layer::<V> { links }
            })
            .collect();

        GraphMem::<V> {
            entry_point: new_entry_point,
            layers: new_layers,
        }
    }

    pub async fn get_entry_point(&self) -> Option<(V::VectorRef, usize)> {
        self.entry_point
            .as_ref()
            .map(|ep| (ep.point.clone(), ep.layer))
    }

    pub async fn set_entry_point(&mut self, point: V::VectorRef, layer: usize) {
        if let Some(previous) = self.entry_point.as_ref() {
            assert!(
                previous.layer < layer,
                "A new entry point should be on a higher layer than before."
            );
        }

        for _ in self.layers.len()..=layer {
            self.layers.push(Layer::new());
        }

        self.entry_point = Some(EntryPoint { point, layer });
    }

    pub async fn get_links(
        &self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> FurthestQueueV<V> {
        let layer = &self.layers[lc];
        if let Some(links) = layer.get_links(base) {
            links.clone()
        } else {
            FurthestQueue::new()
        }
    }

    /// Set the neighbors of vertex `base` at layer `lc` to `links`.  Requires
    /// that the graph already has been extended to have layer `lc` using the
    /// `set_entry_point` function for an entry point at at least this layer.
    ///
    /// Panics if `lc` is higher than the maximum initialized layer.
    pub async fn set_links(&mut self, base: V::VectorRef, links: FurthestQueueV<V>, lc: usize) {
        let layer = self.layers.get_mut(lc).unwrap();
        layer.set_links(base, links);
    }

    pub async fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[derive(PartialEq, Eq, Default, Clone, Debug, Serialize, Deserialize)]
pub struct Layer<V: VectorStore> {
    /// Map a base vector to its neighbors, including the distance base-neighbor.
    links: HashMap<V::VectorRef, FurthestQueueV<V>>,
}

impl<V: VectorStore> Layer<V> {
    fn new() -> Self {
        Layer {
            links: HashMap::new(),
        }
    }

    pub fn from_links(links: HashMap<V::VectorRef, FurthestQueueV<V>>) -> Self {
        Layer { links }
    }

    fn get_links(&self, from: &V::VectorRef) -> Option<&FurthestQueueV<V>> {
        self.links.get(from)
    }

    fn set_links(&mut self, from: V::VectorRef, links: FurthestQueueV<V>) {
        self.links.insert(from, links);
    }

    pub fn get_links_map(&self) -> &HashMap<V::VectorRef, FurthestQueueV<V>> {
        &self.links
    }
}

#[cfg(test)]
mod tests {
    use aes_prng::AesRng;
    use rand::{RngCore, SeedableRng};
    use serde::{Deserialize, Serialize};

    use crate::hnsw::HnswSearcher;

    use hawk_pack::{
        vector_store::lazy_memory_store::{LazyMemoryStore, PointId},
    };

    use super::*;

    #[derive(Default, Clone, Debug, PartialEq, Eq)]
    pub struct TestStore {
        points: HashMap<usize, Point>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Point {
        /// Whatever encoding of a vector.
        data: u64,
        /// Distinguish between queries that are pending, and those that were ultimately accepted into the vector store.
        is_persistent: bool,
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
    pub struct TestPointId(pub usize);

    impl TestPointId {
        pub fn val(&self) -> usize {
            self.0
        }
    }

    fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    impl VectorStore for TestStore {
        type QueryRef = TestPointId; // Vector ID, pending insertion.
        type VectorRef = TestPointId; // Vector ID, inserted.
        type DistanceRef = u32; // Eager distance representation.

        async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
            // The query is now accepted in the store. It keeps the same ID.
            self.points.get_mut(&query.val()).unwrap().is_persistent = true;
            *query
        }

        async fn eval_distance(
            &mut self,
            query: &Self::QueryRef,
            vector: &Self::VectorRef,
        ) -> Self::DistanceRef {
            // Hamming distance
            let vector_0 = self.points[&query.val()].data;
            let vector_1 = self.points[&vector.val()].data;
            hamming_distance(vector_0, vector_1)
        }

        async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
            *distance == 0
        }

        async fn less_than(
            &mut self,
            distance1: &Self::DistanceRef,
            distance2: &Self::DistanceRef,
        ) -> bool {
            *distance1 < *distance2
        }
    }

    #[tokio::test]
    async fn test_from_another_naive() {
        let mut vector_store = LazyMemoryStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::default();
        let mut rng = AesRng::seed_from_u64(0_u64);

        for raw_query in 0..10 {
            let query = vector_store.prepare_query(raw_query);
            let insertion_layer = searcher.select_layer(&mut rng);
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &mut graph_store, &query, insertion_layer)
                .await;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    set_ep,
                )
                .await;
        }

        let equal_graph_store: GraphMem<LazyMemoryStore> =
            GraphMem::from_another(graph_store.clone(), |v| v, |d| d);
        assert_eq!(graph_store, equal_graph_store);

        let different_graph_store: GraphMem<LazyMemoryStore> =
            GraphMem::from_another(graph_store.clone(), |v| PointId(v.val() * 2), |d| d);
        assert_ne!(graph_store, different_graph_store);
    }

    #[tokio::test]
    async fn test_from_another() {
        let mut vector_store = LazyMemoryStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::default();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let mut point_ids: HashMap<PointId, TestPointId> = HashMap::new();
        let mut distances: HashMap<(PointId, PointId), u32> = HashMap::new();

        for raw_query in 0..10 {
            let query = vector_store.prepare_query(raw_query);
            let insertion_layer = searcher.select_layer(&mut rng);
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &mut graph_store, &query, insertion_layer)
                .await;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    set_ep,
                )
                .await;

            point_ids.insert(query, TestPointId(rng.next_u64() as usize));
            for future_query in 0..10_u64 {
                distances.insert(
                    (query, PointId(future_query as usize)),
                    hamming_distance(raw_query, future_query),
                );
            }
        }

        let new_graph_store: GraphMem<TestStore> =
            GraphMem::from_another(graph_store.clone(), |v| point_ids[&v], |d| distances[&d]);

        let (entry_point, layer) = graph_store.get_entry_point().await.unwrap();
        let (new_entry_point, new_layer) = new_graph_store.get_entry_point().await.unwrap();

        // Check that entry points are correct
        assert_eq!(layer, new_layer);
        assert_eq!(point_ids[&entry_point], new_entry_point);

        let layers = graph_store.get_layers();
        let new_layers = new_graph_store.get_layers();

        for (layer, new_layer) in layers.iter().zip(new_layers.iter()) {
            let links = layer.get_links_map();
            let new_links = new_layer.get_links_map();

            for (point_id, queue) in links.iter() {
                let new_point_id = point_ids[point_id];
                let new_queue_vec = new_links[&new_point_id].to_vec();
                let queue_vec = queue.to_vec();
                for ((neighbor_id, distance), (new_neighbor_id, new_distance)) in
                    queue_vec.iter().zip(new_queue_vec)
                {
                    assert_eq!(point_ids[neighbor_id], new_neighbor_id);
                    assert_eq!(distances[distance], new_distance);
                }
            }
        }
    }
}
