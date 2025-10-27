//! Implementation of a hierarchical graph for use by the HNSW algorithm; based
//! on the `GraphMem` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use super::neighborhood::SortedEdgeIds;
use crate::{
    execution::hawk_main::state_check::SetHash,
    hawkers::ideal_knn_engines::{read_knn_results_from_file, Engine, EngineChoice, KNNResult},
    hnsw::{
        searcher::{ConnectPlan, ConnectPlanLayer},
        vector_store::Ref,
    },
};
use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId};
use itertools::izip;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Display, iter::once, path::PathBuf, str::FromStr, sync::Arc};
use tokio::sync::RwLock;

/// Representation of the entry point of HNSW search in a layered graph.
/// This is a vector reference along with the layer of the graph at which
/// search begins.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct EntryPoint<VectorRef> {
    /// The vector reference of the entry point
    pub point: VectorRef,

    /// The layer at which HNSW search begins
    pub layer: usize,
}

/// An in-memory implementation of an HNSW hierarchical graph.
#[derive(Default, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(bound = "V: Ref + Display + FromStr")]
pub struct GraphMem<V: Ref + Display + FromStr> {
    /// Starting vector and layer for HNSW search
    pub entry_point: Option<EntryPoint<V>>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    pub layers: Vec<Layer<V>>,
}

impl<V: Ref + Display + FromStr> Clone for GraphMem<V> {
    fn clone(&self) -> Self {
        GraphMem {
            entry_point: self.entry_point.clone(),
            layers: self.layers.clone(),
        }
    }
}

impl<V: Ref + Display + FromStr> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_point: None,
            layers: vec![],
        }
    }

    pub fn to_arc(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
    }

    pub fn from_precomputed(entry_point: Option<(V, usize)>, layers: Vec<Layer<V>>) -> Self {
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
    pub async fn insert_apply(&mut self, plan: ConnectPlan<V>) {
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
    async fn connect_apply(&mut self, q: V, lc: usize, plan: ConnectPlanLayer<V>) {
        // Connect all n -> q.
        for (n, links) in izip!(plan.neighbors.iter(), plan.nb_links) {
            self.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        self.set_links(q, plan.neighbors, lc).await;
    }

    pub async fn get_entry_point(&self) -> Option<(V, usize)> {
        self.entry_point
            .as_ref()
            .map(|ep| (ep.point.clone(), ep.layer))
    }

    pub async fn set_entry_point(&mut self, point: V, layer: usize) {
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

    pub async fn get_links(&self, base: &V, lc: usize) -> SortedEdgeIds<V> {
        let layer = &self.layers[lc];
        layer.get_links(base).unwrap_or_default()
    }

    /// Set the neighbors of vertex `base` at layer `lc` to `links`.
    pub async fn set_links(&mut self, base: V, links: SortedEdgeIds<V>, lc: usize) {
        if self.layers.len() < lc + 1 {
            self.layers.resize(lc + 1, Layer::new());
        }
        let layer = self.layers.get_mut(lc).unwrap();
        layer.set_links(base, links);
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn checksum(&self) -> u64 {
        let mut set_hash = SetHash::default();
        set_hash.add_unordered(&self.entry_point);
        for (lc, layer) in self.layers.iter().enumerate() {
            set_hash.add_unordered((lc as u64, layer.set_hash.checksum()));
        }
        set_hash.checksum()
    }
}

impl GraphMem<IrisSerialId> {
    pub fn ideal_from_irises(
        irises: Vec<IrisCode>,
        entry_point: Option<(IrisSerialId, usize)>,
        nodes_for_nonzero_layers: Vec<Vec<IrisSerialId>>,
        filepath: PathBuf, // File containing KNN results for layer 0
        k: usize,
        echoice: EngineChoice, // Engine choice for KNN computation on non-zero layer
        num_threads: usize,
    ) -> Self {
        let zero_layer = {
            let results = read_knn_results_from_file(filepath).unwrap();
            Layer::from_knn_results(results)
        };

        let nonzero_layers = nodes_for_nonzero_layers.into_iter().map(|nodes| {
            let iris_data = nodes
                .iter()
                // note 0-indexing of irises (contrast to store usage)
                .map(|node| (*node, irises[(*node - 1) as usize].clone()))
                .collect::<Vec<_>>();
            Layer::ideal_from_irises(iris_data, k, echoice, num_threads)
        });
        GraphMem::from_precomputed(
            entry_point,
            once(zero_layer).chain(nonzero_layers).collect::<Vec<_>>(),
        )
    }
}

#[derive(PartialEq, Eq, Default, Debug, Serialize, Deserialize)]
#[serde(bound = "V: Ref + Display + FromStr")]
pub struct Layer<V: Ref + Display + FromStr> {
    /// Map a base vector to its neighbors, including the distance between
    /// base and neighbor.
    pub links: HashMap<V, SortedEdgeIds<V>>,
    set_hash: SetHash,
}

impl<V: Ref + Display + FromStr> Clone for Layer<V> {
    fn clone(&self) -> Self {
        Layer {
            links: self.links.clone(),
            set_hash: self.set_hash.clone(),
        }
    }
}

impl<V: Ref + Display + FromStr> Layer<V> {
    pub fn new() -> Self {
        Layer {
            links: HashMap::new(),
            set_hash: SetHash::default(),
        }
    }

    pub fn get_links(&self, from: &V) -> Option<SortedEdgeIds<V>> {
        self.links.get(from).cloned()
    }

    pub fn set_links(&mut self, from: V, links: SortedEdgeIds<V>) {
        self.set_hash.add_unordered((&from, &links));

        let previous = self.links.insert(from.clone(), links);

        if let Some(previous) = previous {
            self.set_hash.remove((&from, &previous))
        }
    }

    pub fn get_links_map(&self) -> &HashMap<V, SortedEdgeIds<V>> {
        &self.links
    }

    fn from_knn_results(results: Vec<KNNResult<V>>) -> Self {
        let mut ret = Layer::new();
        for KNNResult { node, neighbors } in results {
            ret.set_links(node, SortedEdgeIds(neighbors));
        }
        ret
    }

    /// Constructs a Layer from pairs of (vectorRef, iris) by computing
    /// the ideal K-nearest neighbors for each such entry.
    fn ideal_from_irises(
        iris_data: Vec<(V, IrisCode)>,
        k: usize,
        echoice: EngineChoice,
        num_threads: usize,
    ) -> Self {
        let (vector_refs, irises): (Vec<V>, Vec<IrisCode>) = iris_data.into_iter().unzip();
        let n = irises.len();
        // Initialize the KNN algorithm;
        let mut engine = Engine::init(echoice, irises, k, 1, num_threads);
        // Run the entire computation
        let results = engine
            .compute_chunk(n)
            .into_iter()
            .map(|result| result.map(|i| vector_refs[(i - 1) as usize].clone()))
            .collect::<Vec<_>>();

        Layer::from_knn_results(results)
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
    U: Ref + Display + FromStr,
    V: Ref + Display + FromStr,
    VecMap: Fn(U) -> V + Copy,
{
    let new_entry_point = graph.entry_point.map(|ep| EntryPoint {
        point: vector_map(ep.point),
        layer: ep.layer,
    });

    let new_layers: Vec<_> = graph
        .layers
        .into_iter()
        .map(|v| {
            let mut layer = Layer::new();
            for (from, nbhd) in v.links.into_iter() {
                layer.set_links(
                    vector_map(from),
                    SortedEdgeIds::from_ascending_vec(nbhd.0.into_iter().map(vector_map).collect()),
                );
            }
            layer
        })
        .collect();

    GraphMem::<V> {
        entry_point: new_entry_point,
        layers: new_layers,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::BufReader, path::PathBuf, sync::Arc};

    use crate::{
        hawkers::{
            ideal_knn_engines::EngineChoice,
            plaintext_store::{PlaintextStore, PlaintextVectorRef},
        },
        hnsw::{
            graph::layered_graph::migrate, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
            VectorStore,
        },
        py_bindings::plaintext_store::Base64IrisCode,
    };
    use aes_prng::AesRng;
    use eyre::Result;
    use iris_mpc_common::{iris_db::db::IrisDB, vector_id::VectorId, IrisSerialId};
    use rand::seq::SliceRandom;
    use rand::{RngCore, SeedableRng};
    use serde_json::Deserializer;

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
        type QueryRef = usize; // Vector ID, pending insertion.
        type VectorRef = usize; // Vector ID, inserted.
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
        ) -> Result<Self::DistanceRef> {
            // Hamming distance
            let vector_0 = self.points[query].data;
            let vector_1 = self.points[vector].data;
            Ok(hamming_distance(vector_0, vector_1))
        }

        async fn eval_minimal_rotation_distance_batch(
            &mut self,
            _query: &Self::QueryRef,
            _vectors: &[Self::VectorRef],
        ) -> Result<Vec<Self::DistanceRef>> {
            unimplemented!()
        }

        async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
            Ok(*distance == 0)
        }

        async fn less_than(
            &mut self,
            distance1: &Self::DistanceRef,
            distance2: &Self::DistanceRef,
        ) -> Result<bool> {
            Ok(*distance1 < *distance2)
        }
    }

    impl VectorStoreMut for TestStore {
        async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
            // The query is now accepted in the store. It keeps the same ID.
            self.points.get_mut(query).unwrap().is_persistent = true;
            *query
        }

        async fn insert_at(
            &mut self,
            _vector_ref: &Self::VectorRef,
            _query: &Self::QueryRef,
        ) -> Result<Self::VectorRef> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_from_another_naive() -> Result<()> {
        let mut vector_store = PlaintextStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let raw_queries = IrisDB::new_random_rng(10, &mut rng);

        for raw_query in raw_queries.db {
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.select_layer_rng(&mut rng)?;
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
                .await?;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    set_ep,
                )
                .await?;
        }

        let different_graph_store: GraphMem<VectorId> = migrate(graph_store.clone(), |v| {
            VectorId::from_0_index(v.index() * 2)
        });
        assert_ne!(graph_store, different_graph_store);

        Ok(())
    }

    #[test]
    fn test_something() -> Result<()> {
        let k = 320;
        let echoice = EngineChoice::NaiveFHD;
        let num_threads = 3;
        let file = std::fs::File::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("iris-mpc-cpu/data/store.ndjson"),
        )?;
        let reader = BufReader::new(file);

        let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
        let irises = stream.map(|e| (&e.unwrap()).into()).collect::<Vec<_>>();
        let n = irises.len();
        // First layer: 1000 random serial ids from 1 to n
        let mut rng = rand::thread_rng();
        let mut first_layer: Vec<IrisSerialId> = (1..=(n as u32)).collect();
        first_layer.shuffle(&mut rng);
        let first_layer = first_layer.into_iter().take(1000).collect::<Vec<_>>();

        // Second layer: 100 random samples from the first layer
        let mut second_layer = first_layer.clone();
        second_layer.shuffle(&mut rng);
        let second_layer = second_layer.into_iter().take(100).collect::<Vec<_>>();
        let entry = second_layer[0];

        let nodes_for_nonzero_layers = vec![first_layer, second_layer];
        let entry_point = Some((entry, 2));
        let filepath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results.txt");

        let graph = GraphMem::ideal_from_irises(
            irises,
            entry_point,
            nodes_for_nonzero_layers,
            filepath,
            k,
            echoice,
            num_threads,
        );

        assert!(graph.layers[0].links.keys().count() == n);
        Ok(())
    }

    #[tokio::test]
    async fn test_from_another() -> Result<()> {
        let mut vector_store = PlaintextStore::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let mut point_ids_map: HashMap<PlaintextVectorRef, usize> = HashMap::new();

        for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.select_layer_rng(&mut rng)?;
            let (neighbors, set_ep) = searcher
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
                .await?;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    set_ep,
                )
                .await?;

            point_ids_map.insert(inserted, rng.next_u32() as usize);
        }

        let new_graph_store: GraphMem<<TestStore as VectorStore>::VectorRef> =
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

        Ok(())
    }
}
