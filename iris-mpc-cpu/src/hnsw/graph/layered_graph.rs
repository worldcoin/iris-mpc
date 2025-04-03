//! Implementation of a hierarchical graph for use by the HNSW algorithm; based
//! on the `GraphMem` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use super::neighborhood::SortedEdgeIds;
use crate::hnsw::{
    searcher::{ConnectPlanLayerV, ConnectPlanV},
    VectorStore,
};
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

/// Representation of the entry point of HNSW search in a layered graph.
/// This is a vector reference along with the layer of the graph at which
/// search begins.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryPoint<VectorRef> {
    /// The vector reference of the entry point
    pub point: VectorRef,

    /// The layer at which HNSW search begins
    pub layer: usize,
}

/// An in-memory implementation of an HNSW hierarchical graph.
#[derive(Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct GraphMem<V: VectorStore> {
    /// Starting vector and layer for HNSW search
    entry_point: Option<EntryPoint<V::VectorRef>>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    layers: Vec<Layer<V>>,
}

impl<V: VectorStore> Clone for GraphMem<V> {
    fn clone(&self) -> Self {
        GraphMem {
            entry_point: self.entry_point.clone(),
            layers: self.layers.clone(),
        }
    }
}

impl<V: VectorStore> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_point: None,
            layers: vec![],
        }
    }

    pub fn to_arc(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
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

    /// Apply an insertion plan from `HnswSearcher::insert_prepare` to the
    /// graph.
    pub async fn insert_apply(&mut self, plan: ConnectPlanV<V>) {
        // If required, set vector as new entry point
        if plan.set_ep {
            let insertion_layer = plan.layers.len() - 1;
            self.set_entry_point(plan.inserted_vector.clone(), insertion_layer)
                .await;
        }

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_plan) in plan.layers.into_iter().enumerate() {
            self.connect_apply(plan.inserted_vector.clone(), lc, layer_plan)
                .await;
        }
    }

    /// Apply the connections from `HnswSearcher::connect_prepare` to the graph.
    async fn connect_apply(&mut self, q: V::VectorRef, lc: usize, plan: ConnectPlanLayerV<V>) {
        // Connect all n -> q.
        for ((n, _nq), links) in izip!(plan.neighbors.iter(), plan.nb_links) {
            self.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors.edge_ids(), lc).await;
    }
}

impl<V: VectorStore> GraphMem<V> {
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

        if self.layers.len() < layer + 1 {
            self.layers.resize(layer + 1, Layer::new());
        }

        self.entry_point = Some(EntryPoint { point, layer });
    }

    pub async fn get_links(
        &self,
        base: &<V as VectorStore>::VectorRef,
        lc: usize,
    ) -> SortedEdgeIds<V::VectorRef> {
        let layer = &self.layers[lc];
        layer.get_links(base).unwrap_or_default()
    }

    /// Set the neighbors of vertex `base` at layer `lc` to `links`.
    pub async fn set_links(
        &mut self,
        base: V::VectorRef,
        links: SortedEdgeIds<V::VectorRef>,
        lc: usize,
    ) {
        if self.layers.len() < lc + 1 {
            self.layers.resize(lc + 1, Layer::new());
        }
        let layer = self.layers.get_mut(lc).unwrap();
        layer.set_links(base, links);
    }

    pub async fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[derive(PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
pub struct Layer<V: VectorStore> {
    /// Map a base vector to its neighbors, including the distance between
    /// base and neighbor.
    links: HashMap<V::VectorRef, SortedEdgeIds<V::VectorRef>>,
}

impl<V: VectorStore> Clone for Layer<V> {
    fn clone(&self) -> Self {
        Layer {
            links: self.links.clone(),
        }
    }
}

impl<V: VectorStore> Layer<V> {
    fn new() -> Self {
        Layer {
            links: HashMap::new(),
        }
    }

    pub fn from_links(links: HashMap<V::VectorRef, SortedEdgeIds<V::VectorRef>>) -> Self {
        Layer { links }
    }

    fn get_links(&self, from: &V::VectorRef) -> Option<SortedEdgeIds<V::VectorRef>> {
        self.links.get(from).cloned()
    }

    fn set_links(&mut self, from: V::VectorRef, links: SortedEdgeIds<V::VectorRef>) {
        self.links.insert(from, links);
    }

    pub fn get_links_map(&self) -> &HashMap<V::VectorRef, SortedEdgeIds<V::VectorRef>> {
        &self.links
    }
}

/// Convert a `GraphMem` data structure via a direct mapping of vector
/// references, leaving the edge sets associated with the mapped
/// vertices unchanged.
///
/// This could be useful for cases where the representation of the graph
/// vertices or distances is changed, but not the underlying values. For
/// example:
/// - vector ids are re-mapped to remove blank entries left by deletions
pub fn migrate<U, V, VecMap>(graph: GraphMem<U>, vector_map: VecMap) -> GraphMem<V>
where
    U: VectorStore,
    V: VectorStore,
    VecMap: Fn(U::VectorRef) -> V::VectorRef + Copy,
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
                .map(|(v, nbhd)| {
                    (
                        vector_map(v),
                        SortedEdgeIds::from_ascending_vec(
                            nbhd.0.into_iter().map(vector_map).collect(),
                        ),
                    )
                })
                .collect();
            Layer::<V> { links }
        })
        .collect();

    GraphMem::<V> {
        entry_point: new_entry_point,
        layers: new_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hawkers::plaintext_store::{PlaintextStore, PointId},
        hnsw::{vector_store::VectorStoreMut, HnswSearcher},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::{RngCore, SeedableRng};

    #[derive(Default, Clone, Debug, PartialEq, Eq)]
    pub struct TestStore {
        points: HashMap<usize, Point>,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Point {
        /// Whatever encoding of a vector.
        data: u64,
        /// Distinguish between queries that are pending, and those that were
        /// ultimately accepted into the vector store.
        is_persistent: bool,
    }

    fn hamming_distance(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    impl VectorStore for TestStore {
        type QueryRef = PointId; // Vector ID, pending insertion.
        type VectorRef = PointId; // Vector ID, inserted.
        type DistanceRef = u32; // Eager distance representation as fraction.

        async fn vectors_as_queries(
            &mut self,
            vectors: Vec<Self::VectorRef>,
        ) -> Vec<Self::QueryRef> {
            vectors
        }

        async fn eval_distance(
            &mut self,
            query: &Self::QueryRef,
            vector: &Self::VectorRef,
        ) -> Self::DistanceRef {
            // Hamming distance
            let vector_0 = self.points[&(query.0 as usize)].data;
            let vector_1 = self.points[&(vector.0 as usize)].data;
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

    impl VectorStoreMut for TestStore {
        async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
            // The query is now accepted in the store. It keeps the same ID.
            self.points
                .get_mut(&(query.0 as usize))
                .unwrap()
                .is_persistent = true;
            *query
        }
    }

    #[tokio::test]
    async fn test_from_another_naive() {
        let mut vector_store = PlaintextStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::default();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let raw_queries = IrisDB::new_random_rng(10, &mut rng);

        for raw_query in raw_queries.db {
            let query = vector_store.prepare_query(raw_query);
            let insertion_layer = searcher.select_layer(&mut rng);
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
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

        let equal_graph_store: GraphMem<PlaintextStore> = migrate(graph_store.clone(), |v| v);
        assert_eq!(graph_store, equal_graph_store);

        let different_graph_store: GraphMem<PlaintextStore> =
            migrate(graph_store.clone(), |v| PointId(v.0 * 2));
        assert_ne!(graph_store, different_graph_store);
    }

    #[tokio::test]
    async fn test_from_another() {
        let mut vector_store = PlaintextStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::default();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let mut point_ids_map: HashMap<<PlaintextStore as VectorStore>::VectorRef, PointId> =
            HashMap::new();

        for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
            let query = vector_store.prepare_query(raw_query);
            let insertion_layer = searcher.select_layer(&mut rng);
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
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

            point_ids_map.insert(query, PointId(rng.next_u32()));
        }

        let new_graph_store: GraphMem<TestStore> =
            migrate(graph_store.clone(), |v| point_ids_map[&v]);

        let (entry_point, layer) = graph_store.get_entry_point().await.unwrap();
        let (new_entry_point, new_layer) = new_graph_store.get_entry_point().await.unwrap();

        // Check that entry points are correct
        assert_eq!(layer, new_layer);
        assert_eq!(point_ids_map[&entry_point], new_entry_point);

        let layers = graph_store.get_layers();
        let new_layers = new_graph_store.get_layers();

        for (layer, new_layer) in layers.iter().zip(new_layers.iter()) {
            let links = layer.get_links_map();
            let new_links = new_layer.get_links_map();

            for (point_id, queue) in links.iter() {
                let new_point_id = point_ids_map[point_id];
                let new_queue_vec = new_links[&new_point_id].to_vec();
                for (neighbor_id, new_neighbor_id) in queue.iter().zip(new_queue_vec) {
                    assert_eq!(point_ids_map[neighbor_id], new_neighbor_id);
                }
            }
        }
    }
}
