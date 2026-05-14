//! Implementation of a hierarchical graph for use by the HNSW algorithm; based
//! on the `GraphMem` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::{
    execution::hawk_main::state_check::SetHash,
    hawkers::ideal_knn_engines::{read_knn_results_from_file, Engine, EngineChoice, KNNResult},
    hnsw::{
        graph::{mutation::EdgeType, GraphMutation, UpdateEntryPoint},
        searcher::LayerMode,
        vector_store::Ref,
        HnswSearcher,
    },
};

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
use itertools::{izip, Itertools};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{
    ser::{SerializeMap, SerializeStruct, Serializer},
    Deserialize, Serialize,
};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    iter::once,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};
use tokio::sync::RwLock;
use tracing::warn;

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
pub struct GraphMem<V: Ref + Display + FromStr + Ord> {
    /// Entry points for HNSW search.
    ///
    /// If the graph is built by a searcher in `LinearScan` mode, this list will contain all nodes assigned
    /// to an `insertion_level >= max_graph_layer`. The searcher uses `get_temporary_entry_point`
    /// while no such node exists.
    ///
    /// If the graph is built by a searcher in `Standard` or `Bounded` mode this list
    /// will contain a single entry point at any given time, which corresponds to a node
    /// in the highest layer of the graph.
    pub entry_points: Vec<EntryPoint<V>>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    pub layers: Vec<Layer<V>>,
}

impl Display for GraphMem<IrisVectorId> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GraphMem")?;
        let eps_str = self
            .entry_points
            .iter()
            .map(|ep| format!("{}:l{}", ep.point, ep.layer))
            .join(", ");
        writeln!(f, "entry_points: [{eps_str}]")?;
        for (lc, layer) in self.layers.iter().enumerate().rev() {
            writeln!(f, "layer: {lc}")?;
            writeln!(f, "{layer}")?;
        }
        Ok(())
    }
}

impl Display for Layer<IrisVectorId> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut links = self
            .links
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect_vec();
        links.sort_by_key(|(k, _)| *k);
        for (id, l) in links.iter() {
            let links_str = l.iter().map(|nb| format!("{nb}")).join(", ");
            writeln!(f, "| {id} :: {links_str}")?;
        }
        Ok(())
    }
}

impl<V: Ref + Display + FromStr + Ord> Clone for GraphMem<V> {
    fn clone(&self) -> Self {
        GraphMem {
            entry_points: self.entry_points.clone(),
            layers: self.layers.clone(),
        }
    }
}

impl<V: Ref + Display + FromStr + Ord> GraphMem<V> {
    pub fn new() -> Self {
        GraphMem {
            entry_points: vec![],
            layers: vec![],
        }
    }

    pub fn to_arc(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
    }

    pub fn from_precomputed(entry_points: Vec<(V, usize)>, layers: Vec<Layer<V>>) -> Self {
        GraphMem {
            entry_points: entry_points
                .into_iter()
                .map(|ep| EntryPoint {
                    point: ep.0,
                    layer: ep.1,
                })
                .collect::<Vec<_>>(),
            layers,
        }
    }

    pub fn get_layers(&self) -> Vec<Layer<V>> {
        self.layers.clone()
    }

    pub fn get_num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return a deterministically selected temporary entry point for the graph.
    ///
    /// This is currently defined as the vector with minimal id in the top
    /// non-empty layer of the graph, or `None` if the graph is empty.
    ///
    /// This is intended to be used in LinearScan mode while the entry_points
    /// list empty.
    pub fn get_temporary_entry_point(&self) -> Option<(V, usize)> {
        self.layers
            .iter()
            .enumerate()
            .rfind(|(_lc, layer)| !layer.links.is_empty())
            .and_then(|(lc, layer)| layer.links.keys().min().map(|x| (x.clone(), lc)))
    }

    /// Gets the list of entry points.
    /// If this list is empty in LinearScan mode, `get_temporary_entry_point` may be used instead.
    pub fn get_entry_points(&self) -> Option<Vec<V>> {
        let v: Vec<_> = self
            .entry_points
            .iter()
            .map(|ep| ep.point.clone())
            .collect();
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    }

    /// Applies a list of graph mutations to the in-memory graph.
    ///
    /// This updates the graph's entry points set and connects the new vector to its
    /// neighbors as specified in the mutations.
    pub fn insert_apply(&mut self, plan: Vec<GraphMutation<V>>) {
        // Pass 1: apply node-level mutations.
        for mutation in plan.iter() {
            match mutation {
                GraphMutation::RemoveNode { id } => {
                    for layer in &mut self.layers {
                        layer.remove_node(id);
                    }
                    self.entry_points.retain(|ep| &ep.point != id);
                }
                GraphMutation::AddNode {
                    id,
                    height,
                    update_ep,
                } => {
                    match update_ep {
                        UpdateEntryPoint::SetUnique { layer } => {
                            if self.layers.len() < *layer + 1 {
                                self.layers.resize(*layer + 1, Layer::new());
                            }
                            self.entry_points = vec![EntryPoint {
                                point: id.clone(),
                                layer: *layer,
                            }];
                        }
                        UpdateEntryPoint::Append { layer } => {
                            self.entry_points.push(EntryPoint {
                                point: id.clone(),
                                layer: *layer,
                            });
                        }
                        UpdateEntryPoint::False => {}
                    }

                    if self.layers.len() < *height {
                        self.layers.resize(*height, Layer::new());
                    }
                    for layer_idx in 0..*height {
                        self.layers[layer_idx].insert_node(id, Vec::new());
                    }
                }
                GraphMutation::AddEdges { .. } | GraphMutation::RemoveEdges { .. } => {}
            }
        }

        // Pass 2: apply edge-level mutations.
        for mutation in plan.into_iter() {
            match mutation {
                GraphMutation::AddNode { .. } | GraphMutation::RemoveNode { .. } => {}
                GraphMutation::AddEdges {
                    base,
                    layer,
                    neighbors: to_add,
                    edge_type,
                } => {
                    if self.layers.len() < layer + 1 {
                        self.layers.resize(layer + 1, Layer::new());
                    }
                    let layer_mut = &mut self.layers[layer];
                    match edge_type {
                        EdgeType::Base => {
                            if layer_mut.get_links(&base).is_none() {
                                warn!(
                                    "AddEdges(Base): base={:?} missing at layer {layer}; skipping",
                                    base
                                );
                            } else {
                                layer_mut.link_neighbors_to_node(&base, to_add);
                            }
                        }
                        EdgeType::Neighbors => {
                            for target in &to_add {
                                if layer_mut.get_links(target).is_none() {
                                    warn!(
                                        "AddEdges(Neighbors): target={:?} missing at layer {layer} (base={:?}); add_neighbor will no-op for this target",
                                        target, base
                                    );
                                }
                            }
                            layer_mut.link_node_to_neighbors(&base, to_add);
                        }
                        EdgeType::All => {
                            if layer_mut.get_links(&base).is_none() {
                                warn!(
                                    "AddEdges(All): base={:?} missing at layer {layer}; skipping outgoing half",
                                    base
                                );
                            } else {
                                layer_mut.link_neighbors_to_node(&base, to_add.clone());
                            }
                            for target in &to_add {
                                if layer_mut.get_links(target).is_none() {
                                    warn!(
                                        "AddEdges(All): target={:?} missing at layer {layer} (base={:?}); add_neighbor will no-op for this target",
                                        target, base
                                    );
                                }
                            }
                            layer_mut.link_node_to_neighbors(&base, to_add);
                        }
                    }
                }
                GraphMutation::RemoveEdges {
                    base,
                    layer,
                    neighbors: to_remove,
                    edge_type,
                } => {
                    if self.layers.len() < layer + 1 {
                        warn!(
                            "RemoveEdges: layer {layer} does not exist (base={:?}); skipping",
                            base
                        );
                        continue;
                    }
                    let layer_mut = &mut self.layers[layer];
                    match edge_type {
                        EdgeType::Base => {
                            if layer_mut.get_links(&base).is_none() {
                                warn!(
                                    "RemoveEdges(Base): base={:?} missing at layer {layer}; skipping",
                                    base
                                );
                            } else {
                                layer_mut.unlink_neighbors_from_node(&base, to_remove);
                            }
                        }
                        EdgeType::Neighbors => {
                            for target in &to_remove {
                                if layer_mut.get_links(target).is_none() {
                                    warn!(
                                        "RemoveEdges(Neighbors): target={:?} missing at layer {layer} (base={:?}); remove_incoming_edges will no-op for this target",
                                        target, base
                                    );
                                }
                            }
                            layer_mut.unlink_node_from_neighbors(&base, to_remove);
                        }
                        EdgeType::All => {
                            if layer_mut.get_links(&base).is_none() {
                                warn!(
                                    "RemoveEdges(All): base={:?} missing at layer {layer}; skipping outgoing half",
                                    base
                                );
                            } else {
                                layer_mut.unlink_neighbors_from_node(&base, to_remove.clone());
                            }
                            for target in &to_remove {
                                if layer_mut.get_links(target).is_none() {
                                    warn!(
                                        "RemoveEdges(All): target={:?} missing at layer {layer} (base={:?}); remove_incoming_edges will no-op for this target",
                                        target, base
                                    );
                                }
                            }
                            layer_mut.unlink_node_from_neighbors(&base, to_remove);
                        }
                    }
                }
            }
        }
    }

    pub async fn get_first_entry_point(&self) -> Option<(V, usize)> {
        self.entry_points
            .first()
            .map(|ep| (ep.point.clone(), ep.layer))
    }

    pub async fn init_entry_points(&mut self, points: Vec<V>, layer: usize) {
        self.entry_points = points
            .into_iter()
            .map(|point| EntryPoint { point, layer })
            .collect()
    }

    pub async fn add_entry_point(&mut self, point: V, layer: usize) {
        if let Some(previous) = self.entry_points.first() {
            assert!(previous.layer == layer, "add_entry_point: layer mismatch");
        }
        self.entry_points.push(EntryPoint { point, layer });
    }

    pub async fn set_unique_entry_point(&mut self, point: V, layer: usize) {
        if let Some(previous) = self.entry_points.first() {
            assert!(
                previous.layer < layer,
                "A new entry point should be on a higher layer than before."
            );
        }

        if self.layers.len() < layer + 1 {
            self.layers.resize(layer + 1, Layer::new());
        }

        self.entry_points = vec![EntryPoint { point, layer }];
    }

    pub async fn get_links(&self, base: &V, lc: usize) -> &[V] {
        let layer = &self.layers[lc];
        layer.get_links(base).unwrap_or(&[])
    }

    /// Set the neighbors of vertex `base` at layer `lc` to `links`.
    pub async fn set_links(&mut self, base: V, links: Vec<V>, lc: usize) {
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
        // Fold entry points in an order-agnostic way: each EntryPoint is hashed
        // individually with a fixed key so a re-ordered `entry_points` Vec still
        // yields the same checksum across parties.
        set_hash.add_unordered_set("entry_points", self.entry_points.iter());
        for (lc, layer) in self.layers.iter().enumerate() {
            set_hash.add_unordered((lc as u64, layer.set_hash.checksum()));
        }
        set_hash.checksum()
    }
}

impl GraphMem<IrisVectorId> {
    /// Builds an idealized GraphMem, where all nearest-neighborhoods are exact.
    ///
    /// Layer 0 is built directly from a file (which generally is expensive to produce).
    /// Nodes in Layer 0 will have `searcher.params.get_M_max(0)` neighbors, if this value is at
    /// most equal to the number of neighbors per node found in the target file. Otherwise, the method panics.
    ///
    /// Higher layers are built from brute-force pairwise computations among all resident nodes. Layer `lc`
    /// will have `searcher.params.get_M_max(lc)`, but consider that the graph might have a Linear-Scan layer.
    ///
    /// The searcher also computes the insertion layers for all nodes (using `prf_seed` for reproducibility).
    /// The engine choice specifies the used distance (FHD or MinFHD).
    pub fn ideal_from_irises(
        irises: Vec<IrisCode>,
        filepath: PathBuf, // File containing KNN results for layer 0
        searcher: &HnswSearcher,
        prf_seed: [u8; 16],
        echoice: EngineChoice, // Engine choice for KNN computation on non-zero layer
    ) -> Result<Self> {
        let zero_layer = {
            let mut results = read_knn_results_from_file(filepath).unwrap();
            for result in results.iter_mut() {
                result.truncate(searcher.params.get_M_max(0));
            }
            let results = results
                .into_par_iter()
                .map(|result| result.map(IrisVectorId::from_serial_id))
                .collect::<Vec<_>>();
            Layer::from_knn_results(results, irises.len())
        };

        let irises_with_vector_ids = izip!(
            zero_layer.links.keys().cloned().sorted(),
            irises.into_iter(),
        )
        .collect::<Vec<_>>();

        // Collect nodes into layers they are inserted into (for layers > 0)
        let mut nonzero_layers_map: BTreeMap<usize, Vec<(IrisVectorId, IrisCode)>> =
            BTreeMap::new();
        for (vector_id, iris) in irises_with_vector_ids.iter() {
            let layer = searcher.gen_layer_prf(&prf_seed, &vector_id)?;
            // Insert node into layers 1 to insertion layer (or not if inserted in layer 0)
            for l in 1..=layer {
                nonzero_layers_map
                    .entry(l)
                    .or_default()
                    .push((*vector_id, iris.clone()));
            }
        }

        let mut nodes_for_nonzero_layers: Vec<Vec<(IrisVectorId, IrisCode)>> =
            nonzero_layers_map.into_values().collect::<Vec<Vec<_>>>();

        // Initialize entry points and truncate layers depending on the layer mode
        let entry_points = match searcher.layer_mode {
            LayerMode::Standard { max_graph_layer } => {
                if let Some(max_layer) = max_graph_layer {
                    nodes_for_nonzero_layers.truncate(max_layer);
                }

                // Entry point is the first vector of the highest non-empty layer, or no
                // entry point if the graph is empty
                once(&irises_with_vector_ids)
                    .chain(nodes_for_nonzero_layers.iter())
                    .last()
                    .unwrap_or(&vec![])
                    .first()
                    .map(|(v, _)| vec![(*v, nodes_for_nonzero_layers.len())])
                    .unwrap_or_default()
            }
            LayerMode::LinearScan { max_graph_layer } => {
                // Entry points are the nodes inserted at the layer after `max_graph_layer`,
                // found at index `max_graph_layer` in the list of nonzero layers
                let entry_points = nodes_for_nonzero_layers
                    .get(max_graph_layer)
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|(v, _)| (*v, max_graph_layer))
                    .collect();

                nodes_for_nonzero_layers.truncate(max_graph_layer);

                entry_points
            }
        };

        // Finally, run brute force algorithms to connect nodes in each layer
        let nonzero_layers =
            nodes_for_nonzero_layers
                .into_iter()
                .enumerate()
                .map(|(i, layer_iris_data)| {
                    Layer::ideal_from_irises(
                        layer_iris_data,
                        searcher.params.get_M_max(i + 1),
                        echoice,
                    )
                });

        Ok(GraphMem::from_precomputed(
            entry_points,
            once(zero_layer).chain(nonzero_layers).collect::<Vec<_>>(),
        ))
    }
}

#[derive(PartialEq, Eq, Default, Debug, Deserialize)]
#[serde(bound = "V: Ref + Display + FromStr")]
pub struct Layer<V: Ref + Display + FromStr + Ord> {
    /// Map a base vector to its neighbors.
    pub links: HashMap<V, Vec<V>>,
    /// A checksum of the layer's links, used for state verification.
    /// This hash is updated whenever links are modified.
    set_hash: SetHash,
}

struct SortedLinks<'a, V: Ord> {
    links: &'a HashMap<V, Vec<V>>,
}

impl<'a, V> Serialize for SortedLinks<'a, V>
where
    V: Serialize + Ord,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut entries: Vec<_> = self.links.iter().collect();
        entries.sort_by_key(|(left, _)| *left);

        let mut map = serializer.serialize_map(Some(entries.len()))?;
        for (key, value) in entries {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

impl<V> Serialize for Layer<V>
where
    V: Ref + Display + FromStr + Ord + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Layer", 2)?;
        state.serialize_field("links", &SortedLinks { links: &self.links })?;
        state.serialize_field("set_hash", &self.set_hash)?;
        state.end()
    }
}

impl<V: Ref + Display + FromStr + Ord> Clone for Layer<V> {
    fn clone(&self) -> Self {
        Layer {
            links: self.links.clone(),
            set_hash: self.set_hash.clone(),
        }
    }
}

impl<V: Ref + Display + FromStr + Ord> Layer<V> {
    pub fn new() -> Self {
        Layer {
            links: HashMap::new(),
            set_hash: SetHash::default(),
        }
    }

    pub fn get_links(&self, from: &V) -> Option<&[V]> {
        self.links.get(from).map(|v| v.as_slice())
    }

    /// Order-agnostic checksum of this layer's link map. Two layers with the
    /// same `(node, set-of-neighbors)` content produce the same checksum even
    /// if their `HashMap` iteration order or internal `Vec` ordering differ.
    pub fn checksum(&self) -> u64 {
        self.set_hash.checksum()
    }

    pub fn insert_node(&mut self, id: &V, neighbors: Vec<V>) {
        self.set_links(id.clone(), neighbors.clone());
    }

    /// Insert `id` as an incoming edge into each target's neighbor list,
    /// keeping each list sorted and deduplicated. Targets that don't exist in
    /// this layer are silently skipped (callers that need to log the missing
    /// case should check `get_links(target)` first). Idempotent: if `id` is
    /// already present in a target's list, that target is left unchanged.
    pub fn link_node_to_neighbors(&mut self, node: &V, neighbors: Vec<V>) {
        for target in &neighbors {
            if let Some(target_links) = self.links.get_mut(target) {
                if let Err(pos) = target_links.binary_search(node) {
                    self.set_hash
                        .remove_unordered_set(target, target_links.iter());
                    target_links.insert(pos, node.clone());
                    self.set_hash.add_unordered_set(target, target_links.iter());
                }
            }
        }
    }

    /// Add `to_add` into `id`'s own neighbor list, sorted and deduplicated.
    /// No-op if `id` is not present in this layer (callers that need to log
    /// the missing case should check `get_links(id)` first). Idempotent:
    /// existing entries are not duplicated.
    pub fn link_neighbors_to_node(&mut self, node: &V, neighbors: Vec<V>) {
        let Some(node_links) = self.links.get_mut(node) else {
            return;
        };
        self.set_hash.remove_unordered_set(node, node_links.iter());
        for nb in neighbors {
            if let Err(pos) = node_links.binary_search(&nb) {
                node_links.insert(pos, nb);
            }
        }
        self.set_hash.add_unordered_set(node, node_links.iter());
    }

    /// Remove a node from the graph and clean up all backlinks from its neighbors.
    pub fn remove_node(&mut self, id: &V) {
        // Remove the node's links and get its neighbors
        if let Some(neighbors) = self.links.remove(id) {
            // Update set_hash for removed node
            self.set_hash.remove_unordered_set(id, neighbors.iter());

            // Remove the node from all neighbors' neighborhoods (bidirectional cleanup)
            // note that if this node did compaction then some old neighbors could still have links
            // to this deleted node. that is ok. it is also ok if the following code block is deleted.
            // this is just an opportunistic low-cost cleanup.
            for neighbor in neighbors {
                if let Some(neighbor_links) = self.links.get_mut(&neighbor) {
                    self.set_hash
                        .remove_unordered_set(&neighbor, neighbor_links.iter());
                    neighbor_links.retain(|x| x != id);
                    self.set_hash
                        .add_unordered_set(&neighbor, neighbor_links.iter());
                }
            }
        }
    }

    /// Remove `id` from each target's neighbor list, where `target` ranges over
    /// `to_remove`. Targets that don't exist in this layer are silently skipped
    /// (the caller's apply path is responsible for any logging).
    pub fn unlink_node_from_neighbors(&mut self, node: &V, neighbors: Vec<V>) {
        for target in &neighbors {
            if let Some(target_links) = self.links.get_mut(target) {
                self.set_hash
                    .remove_unordered_set(target, target_links.iter());
                target_links.retain(|x| x != node);
                self.set_hash.add_unordered_set(target, target_links.iter());
            }
        }
    }

    /// Remove each entry in `to_remove` from `id`'s own neighbor list. No-op
    /// if `id` is not present in this layer (callers that need to log the
    /// missing case should check `get_links(id)` first). The removal is
    /// unidirectional: the targets' own link lists are not modified.
    pub fn unlink_neighbors_from_node(&mut self, node: &V, neighbors: Vec<V>) {
        let Some(node_links) = self.links.get_mut(node) else {
            return;
        };
        self.set_hash.remove_unordered_set(node, node_links.iter());
        node_links.retain(|x| !neighbors.contains(x));
        self.set_hash.add_unordered_set(node, node_links.iter());
    }

    pub fn set_links(&mut self, from: V, links: Vec<V>) {
        use std::collections::hash_map::Entry;
        match self.links.entry(from) {
            Entry::Occupied(mut e) => {
                self.set_hash.remove_unordered_set(e.key(), e.get().iter());
                let existing = e.get_mut();
                existing.clear();
                existing.extend(links);
                self.set_hash.add_unordered_set(e.key(), e.get().iter());
            }
            Entry::Vacant(e) => {
                self.set_hash.add_unordered_set(e.key(), links.iter());
                e.insert(links);
            }
        }
    }

    pub fn get_links_map(&self) -> &HashMap<V, Vec<V>> {
        &self.links
    }

    fn from_knn_results(results: Vec<KNNResult<V>>, n: usize) -> Self {
        let mut ret = Layer::new();
        for KNNResult { node, neighbors } in results.into_iter().take(n) {
            ret.set_links(node, neighbors);
        }
        ret
    }

    /// Constructs a Layer from pairs of (vectorRef, iris) by computing
    /// the ideal K-nearest neighbors for each such entry.
    pub fn ideal_from_irises(
        iris_data: Vec<(V, IrisCode)>,
        k: usize,
        echoice: EngineChoice,
    ) -> Self {
        let (vector_refs, irises): (Vec<V>, Vec<IrisCode>) = iris_data.into_iter().unzip();
        let n = irises.len();
        let k = k.min(n - 1);

        // Initialize the KNN algorithm;
        let mut engine = Engine::init(echoice, irises, k, 1);
        // Run the entire computation
        let results = engine
            .compute_chunk(n)
            .into_iter()
            // remap from engine 1-based indices to original vector ids
            .map(|result| result.map(|i| vector_refs[(i - 1) as usize].clone()))
            .collect::<Vec<_>>();

        Layer::from_knn_results(results, n)
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
    U: Ref + Display + FromStr + Ord,
    V: Ref + Display + FromStr + Ord,
    VecMap: Fn(U) -> V + Copy,
{
    let new_entry_point = graph
        .entry_points
        .iter()
        .map(|ep| EntryPoint {
            point: vector_map(ep.point.clone()),
            layer: ep.layer,
        })
        .collect();

    let new_layers: Vec<_> = graph
        .layers
        .into_iter()
        .map(|v| {
            let mut layer = Layer::new();
            for (from, nbhd) in v.links.into_iter() {
                layer.set_links(vector_map(from), nbhd.into_iter().map(vector_map).collect());
            }
            layer
        })
        .collect();

    GraphMem::<V> {
        entry_points: new_entry_point,
        layers: new_layers,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use crate::{
        hawkers::{
            aby3::aby3_store::FhdOps,
            plaintext_store::{PlaintextStore, PlaintextVectorRef},
        },
        hnsw::{
            graph::layered_graph::migrate, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
            SortedNeighborhood, VectorStore,
        },
    };
    use aes_prng::AesRng;
    use eyre::Result;
    use iris_mpc_common::{iris_db::db::IrisDB, vector_id::VectorId, IrisVectorId};

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
        type QueryRef = usize; // Vector ID, pending insertion.
        type VectorRef = usize; // Vector ID, inserted.
        type DistanceRef = u32; // Eager distance representation as fraction.

        async fn vectors_as_queries(
            &mut self,
            vectors: Vec<Self::VectorRef>,
        ) -> Result<Vec<Self::QueryRef>> {
            Ok(vectors)
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
        let mut vector_store = PlaintextStore::<FhdOps>::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let raw_queries = IrisDB::new_random_rng(10, &mut rng);

        for raw_query in raw_queries.db {
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert::<_, SortedNeighborhood<_>>(
                    &mut vector_store,
                    &graph_store,
                    &query,
                    insertion_layer,
                )
                .await?;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    update_ep,
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
    fn test_layer_deterministic_serialize_order() {
        let mut layer_a = super::Layer::new();
        let mut layer_b = super::Layer::new();

        let v1 = IrisVectorId::from_serial_id(1);
        let v2 = IrisVectorId::from_serial_id(2);
        let v3 = IrisVectorId::from_serial_id(3);
        let v4 = IrisVectorId::from_serial_id(4);
        let v5 = IrisVectorId::from_serial_id(5);

        layer_a.set_links(v1, vec![v2, v3]);
        layer_a.set_links(v4, vec![v5]);

        layer_b.set_links(v4, vec![v5]);
        layer_b.set_links(v1, vec![v2, v3]);

        let bytes_a = bincode::serialize(&layer_a).expect("layer_a serialize");
        let bytes_b = bincode::serialize(&layer_b).expect("layer_b serialize");

        assert_eq!(bytes_a, bytes_b);
    }

    #[tokio::test]
    async fn test_from_another() -> Result<()> {
        let mut vector_store = PlaintextStore::<FhdOps>::new();
        let mut graph_store = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut rng = AesRng::seed_from_u64(0_u64);

        let mut point_ids_map: HashMap<PlaintextVectorRef, usize> = HashMap::new();

        for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert::<_, SortedNeighborhood<_>>(
                    &mut vector_store,
                    &graph_store,
                    &query,
                    insertion_layer,
                )
                .await?;
            let inserted = vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut vector_store,
                    &mut graph_store,
                    inserted,
                    neighbors,
                    update_ep,
                )
                .await?;

            point_ids_map.insert(inserted, rng.next_u32() as usize);
        }

        let new_graph_store: GraphMem<<TestStore as VectorStore>::VectorRef> =
            migrate(graph_store.clone(), |v| point_ids_map[&v]);

        let (entry_point, layer) = graph_store.get_first_entry_point().await.unwrap();
        let (new_entry_point, new_layer) = new_graph_store.get_first_entry_point().await.unwrap();

        // Check that entry points are correct
        assert_eq!(layer, new_layer);
        assert_eq!(point_ids_map[&entry_point], new_entry_point);

        let layers = graph_store.get_layers();
        let new_layers = new_graph_store.get_layers();

        for (layer, new_layer) in layers.iter().zip(new_layers.iter()) {
            let links = layer.get_links_map();
            let new_links = new_layer.get_links_map();

            #[allow(
                clippy::iter_over_hash_type,
                reason = "Iteration is for assertions against a parallel data structure, compared entry by entry."
            )]
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

    use crate::hnsw::graph::mutation::{EdgeType, GraphMutation, UpdateEntryPoint};

    #[test]
    fn add_edges_outgoing_writes_only_to_id_list() {
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        let c = IrisVectorId::from_serial_id(3);
        // Seed: a, b, c all exist at layer 0 with no edges.
        graph.insert_apply(vec![
            GraphMutation::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 1 },
            },
            GraphMutation::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
            GraphMutation::AddNode {
                id: c,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
        ]);
        graph.insert_apply(vec![GraphMutation::AddEdges {
            base: a,
            layer: 0,
            neighbors: vec![b, c],
            edge_type: EdgeType::Base,
        }]);
        assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b, c]);
        assert_eq!(
            graph.layers[0].get_links(&b).unwrap(),
            &[] as &[IrisVectorId]
        );
        assert_eq!(
            graph.layers[0].get_links(&c).unwrap(),
            &[] as &[IrisVectorId]
        );
    }

    #[test]
    fn add_edges_incoming_writes_only_to_target_lists() {
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        let c = IrisVectorId::from_serial_id(3);
        graph.insert_apply(vec![
            GraphMutation::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 1 },
            },
            GraphMutation::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
            GraphMutation::AddNode {
                id: c,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
        ]);
        graph.insert_apply(vec![GraphMutation::AddEdges {
            base: a,
            layer: 0,
            neighbors: vec![b, c],
            edge_type: EdgeType::Neighbors,
        }]);
        assert_eq!(
            graph.layers[0].get_links(&a).unwrap(),
            &[] as &[IrisVectorId]
        );
        assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
        assert_eq!(graph.layers[0].get_links(&c).unwrap(), &[a]);
    }

    #[test]
    fn add_edges_bidirectional_writes_both_sides() {
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        let c = IrisVectorId::from_serial_id(3);
        graph.insert_apply(vec![
            GraphMutation::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 1 },
            },
            GraphMutation::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
            GraphMutation::AddNode {
                id: c,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
        ]);
        graph.insert_apply(vec![GraphMutation::AddEdges {
            base: a,
            layer: 0,
            neighbors: vec![b, c],
            edge_type: EdgeType::All,
        }]);
        assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b, c]);
        assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
        assert_eq!(graph.layers[0].get_links(&c).unwrap(), &[a]);
    }

    #[test]
    fn remove_edges_outgoing_only_modifies_id_list() {
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        let c = IrisVectorId::from_serial_id(3);
        graph.insert_apply(vec![
            GraphMutation::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 1 },
            },
            GraphMutation::AddEdges {
                base: a,
                layer: 0,
                neighbors: vec![b, c],
                edge_type: EdgeType::Base,
            },
            GraphMutation::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
            GraphMutation::AddEdges {
                base: b,
                layer: 0,
                neighbors: vec![a],
                edge_type: EdgeType::Base,
            },
            GraphMutation::AddNode {
                id: c,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
            GraphMutation::AddEdges {
                base: c,
                layer: 0,
                neighbors: vec![a],
                edge_type: EdgeType::Base,
            },
        ]);
        graph.insert_apply(vec![GraphMutation::RemoveEdges {
            base: a,
            layer: 0,
            neighbors: vec![b],
            edge_type: EdgeType::Base,
        }]);
        assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[c]);
        // Bidirectional cleanup is not implied — b's list still contains a.
        assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
    }

    #[test]
    fn two_phase_apply_edges_before_node_in_vec_still_works() {
        // Pass 1 should apply AddNode before pass 2 applies AddEdges, regardless
        // of their order in the input Vec.
        let mut graph = GraphMem::<IrisVectorId>::new();
        let a = IrisVectorId::from_serial_id(1);
        let b = IrisVectorId::from_serial_id(2);
        graph.insert_apply(vec![
            // Listed first: an edge op that references a node not yet created.
            GraphMutation::AddEdges {
                base: a,
                layer: 0,
                neighbors: vec![b],
                edge_type: EdgeType::Base,
            },
            // Listed second: the node creation.
            GraphMutation::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::SetUnique { layer: 0 },
            },
            GraphMutation::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            },
        ]);
        // Pass-1 created the nodes, then pass-2 applied the edge — so a should
        // now have b in its outgoing list.
        assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b]);
    }
}
