//! Implementation of a hierarchical graph for use by the HNSW algorithm; based
//! on the `GraphMem` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use crate::{
    execution::hawk_main::state_check::SetHash,
    hawkers::ideal_knn_engines::{
        read_knn_results_from_file, EngineChoice, IdealKnn, KNNResult, NaiveKNN,
    },
    hnsw::{
        graph::{
            mutation::{EdgeType, UnstampedMutation},
            GraphMutation, MutationOp, UpdateEntryPoint,
        },
        HnswSearcher,
    },
};

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, SerialId};
use itertools::{izip, Itertools};
use serde::{
    ser::{SerializeMap, SerializeStruct, Serializer},
    Deserialize, Serialize,
};
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    iter::once,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::RwLock;
use tracing::warn;

/// Capability token carrying the sequence number to stamp on a neighborhood
/// edit. Constructible only within the graph module, so no external code can
/// advance a neighborhood's `seq_no` out of band. Required by
/// [`Layer::edit_links`].
#[derive(Clone, Copy, Debug)]
pub struct Tick(u64);

impl Tick {
    pub(in crate::hnsw::graph) fn new(seq_no: u64) -> Self {
        Tick(seq_no)
    }

    fn value(self) -> u64 {
        self.0
    }
}

/// The neighbor list of a single node in one layer, together with the sequence
/// number of the mutation that last modified it.
///
/// Fields are private: `neighbors` and `seq_no` move together so the
/// "all edges fresh as of `seq_no`" invariant can only be established (and not
/// silently broken) through [`Layer`]'s mutators.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neighborhood {
    neighbors: Vec<SerialId>,
    seq_no: u64,
}

impl Neighborhood {
    /// The neighbor list. Read-only; mutate via [`Layer`].
    pub fn neighbors(&self) -> &[SerialId] {
        &self.neighbors
    }

    /// Sequence number of the mutation that last modified this neighborhood.
    /// Under the freshness invariant, every edge in `neighbors` is valid as of
    /// this tick.
    pub fn seq_no(&self) -> u64 {
        self.seq_no
    }
}

/// Whether edge `A -> z` is valid for a neighborhood last certified at `old_seq`:
/// `z` must be a live node (present in the content clock) *and* its content must
/// not have advanced past `old_seq`. An absent stamp means dead (removed or never
/// added) and a stamp `> old_seq` means content-refreshed (reauthed) since the
/// edge was certified — either way the edge is dropped.
///
/// Used both at write time (the filter-on-bump in `insert_apply`, which drops
/// invalid edges before re-stamping a neighborhood) and at read time
/// ([`GraphMem::get_active_links`], which skips them during traversal).
fn is_active(content: &HashMap<SerialId, u64>, z: SerialId, old_seq: u64) -> bool {
    content.get(&z).is_some_and(|&c| c <= old_seq)
}

/// Representation of the entry point of HNSW search in a layered graph.
/// This is a vector reference along with the layer of the graph at which
/// search begins.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct EntryPoint {
    /// The serial id of the entry point node
    pub point: SerialId,

    /// The layer at which HNSW search begins
    pub layer: usize,
}

/// An in-memory implementation of an HNSW hierarchical graph.
#[derive(Default, PartialEq, Eq, Debug)]
pub struct GraphMem {
    /// Entry points for HNSW search.
    ///
    /// This list contains all nodes assigned to an `insertion_level >= max_graph_layer`.
    /// The searcher uses `get_temporary_entry_point` while no such node exists.
    pub entry_points: Vec<EntryPoint>,

    /// The layers of the hierarchical graph. The nodes of each layer are a
    /// subset of the nodes of the previous layer, and graph neighborhoods in
    /// each layer represent approximate nearest neighbors within that layer.
    pub layers: Vec<Layer>,

    /// The sequence number of the most recently applied `GraphMutation`. `0`
    /// means no mutation has been applied. Advanced by `insert_apply` on
    /// success.
    pub last_update_seq_no: u64,

    /// The sequence number of the mutation that last touched each node (either
    /// inserting it or modifying one of its edge lists). Removed when the node
    /// is deleted.
    pub node_init_seq_no: HashMap<SerialId, u64>,

    /// Incrementally-maintained order-agnostic hash of `node_init_seq_no`,
    /// folded into [`GraphMem::checksum`] so the content clock is part of
    /// cross-party consensus (it gates edge liveness via `is_active`). Derived
    /// state: not serialized; recomputed from `node_init_seq_no` on construction
    /// and deserialization, kept in sync by `insert_apply`. Private so all
    /// construction routes through `from_parts`, which computes it.
    node_init_hash: SetHash,
}

impl Display for GraphMem {
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

impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut links = self
            .links
            .iter()
            .map(|(k, v)| (*k, v.neighbors.clone()))
            .collect_vec();
        links.sort_by_key(|(k, _)| *k);
        for (id, l) in links.iter() {
            let links_str = l.iter().map(|nb| format!("{nb}")).join(", ");
            writeln!(f, "| {id} :: {links_str}")?;
        }
        Ok(())
    }
}

impl Clone for GraphMem {
    fn clone(&self) -> Self {
        GraphMem {
            entry_points: self.entry_points.clone(),
            layers: self.layers.clone(),
            last_update_seq_no: self.last_update_seq_no,
            node_init_seq_no: self.node_init_seq_no.clone(),
            node_init_hash: self.node_init_hash.clone(),
        }
    }
}

impl<'de> Deserialize<'de> for GraphMem {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Mirror the field order of the manual `Serialize` impl; `node_init_hash`
        // is derived and recomputed by `from_parts` rather than read.
        #[derive(Deserialize)]
        struct GraphMemData {
            entry_points: Vec<EntryPoint>,
            layers: Vec<Layer>,
            last_update_seq_no: u64,
            node_init_seq_no: HashMap<SerialId, u64>,
        }
        let d = GraphMemData::deserialize(deserializer)?;
        Ok(GraphMem::from_parts(
            d.entry_points,
            d.layers,
            d.last_update_seq_no,
            d.node_init_seq_no,
        ))
    }
}

impl GraphMem {
    pub fn new() -> Self {
        GraphMem {
            entry_points: vec![],
            layers: vec![],
            last_update_seq_no: 0,
            node_init_seq_no: HashMap::new(),
            node_init_hash: SetHash::default(),
        }
    }

    /// Assemble a `GraphMem` from its parts, computing the derived
    /// `node_init_hash`. The sole constructor for callers outside this module
    /// (the field is private), so the content-clock hash can never be missed.
    pub fn from_parts(
        entry_points: Vec<EntryPoint>,
        layers: Vec<Layer>,
        last_update_seq_no: u64,
        node_init_seq_no: HashMap<SerialId, u64>,
    ) -> Self {
        let mut node_init_hash = SetHash::default();
        // SetHash folds via commutative wrapping addition, so iteration order is
        // irrelevant to the result.
        #[allow(clippy::iter_over_hash_type)]
        for (&serial, &seq) in &node_init_seq_no {
            node_init_hash.add_unordered(Self::node_init_contribution(serial, seq));
        }
        GraphMem {
            entry_points,
            layers,
            last_update_seq_no,
            node_init_seq_no,
            node_init_hash,
        }
    }

    /// Single source of truth for how a `(serial, seq)` content-clock entry is
    /// folded into `node_init_hash` — used by both the bulk build in
    /// `from_parts` and the incremental updates in `insert_apply`, so the two
    /// can never drift.
    fn node_init_contribution(serial: SerialId, seq: u64) -> (&'static str, SerialId, u64) {
        ("node_init", serial, seq)
    }

    /// Returns the sequence number that the next applied `GraphMutation` must
    /// equal or exceed. This is a pure peek — it does not modify the graph.
    pub fn next_sequence_number(&self) -> u64 {
        self.last_update_seq_no + 1
    }

    pub fn to_arc(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
    }

    pub fn from_precomputed(entry_points: Vec<(SerialId, usize)>, layers: Vec<Layer>) -> Self {
        // Seed the content clock at 0 for every node so the read-path liveness
        // filter (`is_active`) treats these trusted nodes as live; without this
        // `get_active_links` would drop every edge.
        let node_init_seq_no = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| (v, 0u64))
            .collect();
        let entry_points = entry_points
            .into_iter()
            .map(|ep| EntryPoint {
                point: ep.0,
                layer: ep.1,
            })
            .collect::<Vec<_>>();
        GraphMem::from_parts(entry_points, layers, 0, node_init_seq_no)
    }

    pub fn get_layers(&self) -> Vec<Layer> {
        self.layers.clone()
    }

    pub fn get_num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Return a deterministically selected temporary entry point for the graph.
    ///
    /// This is currently defined as the node with the minimal serial id in the
    /// top non-empty layer of the graph, or `None` if the graph is empty.
    ///
    /// This is intended to be used in LinearScan mode while the entry_points
    /// list is empty.
    pub fn get_temporary_entry_point(&self) -> Option<(SerialId, usize)> {
        self.layers
            .iter()
            .enumerate()
            .rfind(|(_lc, layer)| !layer.links.is_empty())
            .and_then(|(lc, layer)| layer.links.keys().min().map(|x| (*x, lc)))
    }

    /// Gets the list of entry points.
    /// If this list is empty in LinearScan mode, `get_temporary_entry_point` may be used instead.
    pub fn get_entry_points(&self) -> Option<Vec<SerialId>> {
        let v: Vec<_> = self.entry_points.iter().map(|ep| ep.point).collect();
        if v.is_empty() {
            None
        } else {
            Some(v)
        }
    }

    /// Applies a list of graph mutations to the in-memory graph.
    ///
    /// This updates the graph's entry points set and connects the new vector to
    /// its neighbors as specified in the mutations.
    ///
    /// The supplied `mutation.seq_no` must be strictly greater than
    /// `self.last_update_seq_no`; otherwise the call returns `Err` without
    /// touching the graph. On success `self.last_update_seq_no` advances to
    /// `mutation.seq_no`.
    pub fn insert_apply(&mut self, mutation: &GraphMutation) -> Result<()> {
        if mutation.seq_no <= self.last_update_seq_no {
            return Err(eyre::eyre!(
                "GraphMem::insert_apply: mutation seq_no {} is not strictly greater than \
                 last_update_seq_no {}",
                mutation.seq_no,
                self.last_update_seq_no,
            ));
        }

        let seq_no = mutation.seq_no;

        // Pass 1: apply node-level mutations.
        for op in mutation.ops.iter() {
            match op {
                MutationOp::RemoveNode { id } => {
                    let sid = id.serial_id();
                    for layer in &mut self.layers {
                        layer.remove_node(sid);
                    }
                    self.entry_points.retain(|ep| ep.point != sid);
                    if let Some(old) = self.node_init_seq_no.remove(&sid) {
                        self.node_init_hash
                            .remove(Self::node_init_contribution(sid, old));
                    }
                }
                MutationOp::AddNode {
                    id,
                    height,
                    update_ep,
                } => {
                    let sid = id.serial_id();
                    match update_ep {
                        UpdateEntryPoint::Append { layer } => {
                            self.entry_points.push(EntryPoint {
                                point: sid,
                                layer: *layer,
                            });
                        }
                        UpdateEntryPoint::False => {}
                    }

                    if self.layers.len() < *height {
                        self.layers.resize(*height, Layer::new());
                    }
                    for layer_idx in 0..*height {
                        self.layers[layer_idx].insert_node(sid, Vec::new(), seq_no);
                    }
                    if let Some(old) = self.node_init_seq_no.insert(sid, seq_no) {
                        self.node_init_hash
                            .remove(Self::node_init_contribution(sid, old));
                    }
                    self.node_init_hash
                        .add_unordered(Self::node_init_contribution(sid, seq_no));
                }
                MutationOp::AddEdges { .. } | MutationOp::RemoveEdges { .. } => {}
            }
        }

        // Pass 2: apply edge-level mutations. Each touched neighborhood drops its
        // invalid edges (see `is_active`) then is re-stamped in one step (see
        // `Layer::edit_links`): the drop happens *before* any new edge is
        // appended, so an append can't re-certify an already-invalid sibling as
        // fresh. Edge ops never advance the content clock — only a node's own
        // (re-)insertion does.
        let tick = Tick::new(seq_no);
        for op in mutation.ops.iter() {
            match op {
                MutationOp::AddNode { .. } | MutationOp::RemoveNode { .. } => {}
                MutationOp::AddEdges {
                    base,
                    layer,
                    neighbors: to_add,
                    edge_type,
                } => {
                    let layer = *layer;
                    if self.layers.len() < layer + 1 {
                        self.layers.resize(layer + 1, Layer::new());
                    }
                    let base_sid = base.serial_id();
                    let neighbor_sids: Vec<SerialId> =
                        to_add.iter().map(|v| v.serial_id()).collect();
                    let content = &self.node_init_seq_no;
                    let layer_mut = &mut self.layers[layer];
                    // Forward half: append `neighbor_sids` to base's own list.
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        if layer_mut.get_links(&base_sid).is_none() {
                            warn!("AddEdges({edge_type:?}): base={base:?} missing at layer {layer}; skipping outgoing half");
                        } else {
                            layer_mut.edit_links(base_sid, tick, |old_seq, nbrs| {
                                nbrs.retain(|z| is_active(content, *z, old_seq));
                                nbrs.extend_from_slice(&neighbor_sids);
                            });
                        }
                    }
                    // Back half: append base into each target's list.
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for (target, target_sid) in to_add.iter().zip(neighbor_sids.iter()) {
                            if layer_mut.get_links(target_sid).is_none() {
                                warn!("AddEdges({edge_type:?}): target={target:?} missing at layer {layer} (base={base:?}); skipping back-edge");
                            } else {
                                layer_mut.edit_links(*target_sid, tick, |old_seq, nbrs| {
                                    nbrs.retain(|z| is_active(content, *z, old_seq));
                                    nbrs.push(base_sid);
                                });
                            }
                        }
                    }
                }
                MutationOp::RemoveEdges {
                    base,
                    layer,
                    neighbors: to_remove,
                    edge_type,
                } => {
                    let layer = *layer;
                    if self.layers.len() < layer + 1 {
                        warn!(
                            "RemoveEdges: layer {layer} does not exist (base={base:?}); skipping"
                        );
                        continue;
                    }
                    let base_sid = base.serial_id();
                    let remove_sids: Vec<SerialId> =
                        to_remove.iter().map(|v| v.serial_id()).collect();
                    let content = &self.node_init_seq_no;
                    let layer_mut = &mut self.layers[layer];
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        if layer_mut.get_links(&base_sid).is_none() {
                            warn!("RemoveEdges({edge_type:?}): base={base:?} missing at layer {layer}; skipping outgoing half");
                        } else {
                            layer_mut.edit_links(base_sid, tick, |old_seq, nbrs| {
                                nbrs.retain(|z| {
                                    is_active(content, *z, old_seq) && !remove_sids.contains(z)
                                });
                            });
                        }
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for (target, target_sid) in to_remove.iter().zip(remove_sids.iter()) {
                            if layer_mut.get_links(target_sid).is_none() {
                                warn!("RemoveEdges({edge_type:?}): target={target:?} missing at layer {layer} (base={base:?}); skipping");
                            } else {
                                layer_mut.edit_links(*target_sid, tick, |old_seq, nbrs| {
                                    nbrs.retain(|z| {
                                        is_active(content, *z, old_seq) && *z != base_sid
                                    });
                                });
                            }
                        }
                    }
                }
            }
        }

        self.last_update_seq_no = mutation.seq_no;
        Ok(())
    }

    /// Stamp a locally-built [`UnstampedMutation`] with the next sequence
    /// number, apply it, and return the resulting [`GraphMutation`].
    ///
    /// This is the sole minter of sequence numbers for in-process mutations:
    /// the number is assigned from `next_sequence_number()` and consumed by the
    /// apply in one step, so `last_update_seq_no` can never lag behind the
    /// highest minted id and two mutations can never share a number. Mutations
    /// that already carry a sequence number (replayed from a WAL or checkpoint)
    /// go through `insert_apply`/`insert_apply_all` instead, where the strict
    /// monotonicity check guards the externally-supplied `seq_no`.
    pub fn apply_new(&mut self, mutation: UnstampedMutation) -> Result<GraphMutation> {
        let stamped = GraphMutation {
            seq_no: self.next_sequence_number(),
            ops: mutation.ops,
        };
        self.insert_apply(&stamped)?;
        Ok(stamped)
    }

    pub fn insert_apply_all(&mut self, mutations: &[GraphMutation]) -> Result<()> {
        for m in mutations {
            self.insert_apply(m)?;
        }
        Ok(())
    }

    pub async fn get_first_entry_point(&self) -> Option<(SerialId, usize)> {
        self.entry_points.first().map(|ep| (ep.point, ep.layer))
    }

    pub async fn init_entry_points(&mut self, points: Vec<SerialId>, layer: usize) {
        self.entry_points = points
            .into_iter()
            .map(|point| EntryPoint { point, layer })
            .collect()
    }

    pub async fn add_entry_point(&mut self, point: SerialId, layer: usize) {
        if let Some(previous) = self.entry_points.first() {
            assert!(previous.layer == layer, "add_entry_point: layer mismatch");
        }
        self.entry_points.push(EntryPoint { point, layer });
    }

    pub async fn set_unique_entry_point(&mut self, point: SerialId, layer: usize) {
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

    /// The stored neighbor list of `base` at layer `lc`, verbatim — no staleness
    /// filtering. For maintenance paths (compaction, pruning) that reason about
    /// physical edges; search traversal must use [`Self::get_active_links`].
    /// Empty if `base`/`lc` absent.
    pub async fn get_raw_links(&self, base: &SerialId, lc: usize) -> &[SerialId] {
        let layer = &self.layers[lc];
        layer
            .get_links(base)
            .map(|n| n.neighbors.as_slice())
            .unwrap_or(&[])
    }

    /// Neighbors of `base` at layer `lc` valid for traversal: a neighbor `z` is
    /// kept iff it is [`is_active`] (live and not content-refreshed past this
    /// neighborhood's last-certified `seq_no`). Applies the same filter as the
    /// write-path filter-on-bump, so traversal skips reauthed and removed edges
    /// even in neighborhoods a write has not yet physically cleaned. Unlike
    /// [`Self::get_raw_links`] it returns an owned `Vec`. Empty if `base`/`lc`
    /// absent.
    pub fn get_active_links(&self, base: &SerialId, lc: usize) -> Vec<SerialId> {
        let Some(nbhd) = self.layers.get(lc).and_then(|layer| layer.get_links(base)) else {
            return Vec::new();
        };
        let old_seq = nbhd.seq_no;
        nbhd.neighbors
            .iter()
            .copied()
            .filter(|z| is_active(&self.node_init_seq_no, *z, old_seq))
            .collect()
    }

    /// Set the neighbors of vertex `base` at layer `lc` to `links`.
    pub async fn set_links(
        &mut self,
        base: SerialId,
        links: Vec<SerialId>,
        lc: usize,
        seq_no: u64,
    ) {
        if self.layers.len() < lc + 1 {
            self.layers.resize(lc + 1, Layer::new());
        }
        let layer = self.layers.get_mut(lc).unwrap();
        layer.set_links(base, links, seq_no);
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
            // Each layer's `set_hash` now folds (node, seq_no, neighbors), so the
            // per-neighborhood freshness clock is covered here.
            set_hash.add_unordered((lc as u64, layer.set_hash.checksum()));
        }
        // Fold the content clock (node_init_seq_no) so parties that would skip
        // different edges via `is_active` disagree on the checksum.
        set_hash.add_unordered(("node_init_clock", self.node_init_hash.checksum()));
        set_hash.checksum()
    }
}

impl GraphMem {
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
        use crate::hawkers::{
            aby3::aby3_store::{FhdOps, NhdOps},
            ideal_knn_engines::IrisKnn,
        };
        match echoice {
            EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => ideal_graph_from_vectors(
                irises,
                filepath,
                searcher,
                prf_seed,
                IrisKnn::<FhdOps>::new(echoice.distance_mode()),
            ),
            EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => ideal_graph_from_vectors(
                irises,
                filepath,
                searcher,
                prf_seed,
                IrisKnn::<NhdOps>::new(echoice.distance_mode()),
            ),
        }
    }

    /// Idealized GraphMem for deep-ID Int4 vectors. Layer 0 is loaded from a
    /// pre-computed KNN file (same format used by `ideal_from_irises`); higher
    /// layers are brute-forced using inner-product (greater dot = closer).
    pub fn ideal_from_int4_vectors(
        vectors: Vec<crate::hawkers::plaintext_deep_id_store::Int4Vector>,
        filepath: PathBuf,
        searcher: &HnswSearcher,
        prf_seed: [u8; 16],
        echoice: crate::hawkers::ideal_knn_engines::EngineChoiceInt4,
    ) -> Result<Self> {
        use crate::hawkers::ideal_knn_engines::{EngineChoiceInt4, Int4DotKnn};
        match echoice {
            EngineChoiceInt4::NaiveInt4Dot => ideal_graph_from_vectors::<Int4DotKnn>(
                vectors, filepath, searcher, prf_seed, Int4DotKnn,
            ),
        }
    }
}

/// Shared body for `ideal_from_irises` / `ideal_from_int4_vectors`: builds an
/// idealized hierarchical graph where layer 0 is loaded from a pre-computed
/// KNN file and higher layers are brute-forced via `NaiveKNN<K>`.
fn ideal_graph_from_vectors<K>(
    vectors: Vec<K::Vector>,
    filepath: PathBuf,
    searcher: &HnswSearcher,
    prf_seed: [u8; 16],
    knn_proto: K,
) -> Result<GraphMem>
where
    K: IdealKnn,
    K::Vector: Clone,
{
    let zero_layer = {
        let mut results = read_knn_results_from_file(filepath)?;
        for result in results.iter_mut() {
            result.truncate(searcher.params.get_M_max(0));
        }
        Layer::from_knn_results(results, vectors.len())
    };

    let vectors_with_ids: Vec<(SerialId, K::Vector)> = izip!(
        zero_layer.links.keys().cloned().sorted(),
        vectors.into_iter(),
    )
    .collect();

    // Collect nodes into layers they are inserted into (for layers > 0)
    let mut nonzero_layers_map: BTreeMap<usize, Vec<(SerialId, K::Vector)>> = BTreeMap::new();
    for (serial_id, v) in vectors_with_ids.iter() {
        let layer = searcher.gen_layer_prf(&prf_seed, serial_id)?;
        for l in 1..=layer {
            nonzero_layers_map
                .entry(l)
                .or_default()
                .push((*serial_id, v.clone()));
        }
    }

    let mut nodes_for_nonzero_layers: Vec<Vec<(SerialId, K::Vector)>> =
        nonzero_layers_map.into_values().collect();

    let max_graph_layer = searcher.max_graph_layer;
    let entry_points: Vec<(SerialId, usize)> = nodes_for_nonzero_layers
        .get(max_graph_layer)
        .unwrap_or(&vec![])
        .iter()
        .map(|(v, _)| (*v, max_graph_layer))
        .collect();
    nodes_for_nonzero_layers.truncate(max_graph_layer);

    let nonzero_layers = nodes_for_nonzero_layers
        .into_iter()
        .enumerate()
        .map(|(i, layer_data)| {
            Layer::ideal_from_data::<K>(
                layer_data,
                searcher.params.get_M_max(i + 1),
                knn_proto.clone(),
            )
        });

    Ok(GraphMem::from_precomputed(
        entry_points,
        once(zero_layer).chain(nonzero_layers).collect::<Vec<_>>(),
    ))
}

#[derive(PartialEq, Eq, Default, Debug, Deserialize)]
pub struct Layer {
    /// Map a node's serial id to its neighborhood (neighbors + last-update seq_no).
    pub links: HashMap<SerialId, Neighborhood>,
    /// A checksum of the layer's links, used for state verification.
    /// This hash is updated whenever links are modified.
    set_hash: SetHash,
}

struct SortedLinks<'a> {
    links: &'a HashMap<SerialId, Neighborhood>,
}

impl<'a> Serialize for SortedLinks<'a> {
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

impl Serialize for Layer {
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

impl Serialize for GraphMem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("GraphMem", 4)?;
        state.serialize_field("entry_points", &self.entry_points)?;
        state.serialize_field("layers", &self.layers)?;
        state.serialize_field("last_update_seq_no", &self.last_update_seq_no)?;
        let sorted_node_init: BTreeMap<_, _> = self.node_init_seq_no.iter().collect();
        state.serialize_field("node_init_seq_no", &sorted_node_init)?;
        state.end()
    }
}

impl Clone for Layer {
    fn clone(&self) -> Self {
        Layer {
            links: self.links.clone(),
            set_hash: self.set_hash.clone(),
        }
    }
}

impl Layer {
    pub fn new() -> Self {
        Layer {
            links: HashMap::new(),
            set_hash: SetHash::default(),
        }
    }

    /// Empty layer with the link map pre-sized for `n` nodes, avoiding repeated
    /// rehashing when bulk-loading a known-size layer.
    pub fn with_capacity(n: usize) -> Self {
        Layer {
            links: HashMap::with_capacity(n),
            set_hash: SetHash::default(),
        }
    }

    pub fn get_links(&self, from: &SerialId) -> Option<&Neighborhood> {
        self.links.get(from)
    }

    /// Order-agnostic checksum of this layer's link map. Two layers with the
    /// same `(node, set-of-neighbors)` content produce the same checksum even
    /// if their `HashMap` iteration order or internal `Vec` ordering differ.
    pub fn checksum(&self) -> u64 {
        self.set_hash.checksum()
    }

    pub fn insert_node(&mut self, id: SerialId, neighbors: Vec<SerialId>, seq_no: u64) {
        self.set_links(id, neighbors, seq_no);
    }

    /// Bracketed, filter-then-stamp edit of `node`'s neighbor list — the sole
    /// path that incrementally grows or shrinks an existing neighborhood.
    ///
    /// Removes the old set-hash contribution, runs `f` (which receives the
    /// pre-edit `seq_no` so it can drop entries that became stale relative to
    /// it), then re-sorts, dedups, stamps `tick`, and re-adds the set-hash
    /// contribution. No-op if `node` is absent from this layer.
    ///
    /// Requires a [`Tick`] (constructible only inside the graph module), so no
    /// out-of-band `seq_no` can be stamped. `f` must filter the existing list
    /// *before* pushing new neighbors: freshly added edges are fresh as of
    /// `tick` and must not be subject to the stale filter, whose threshold is
    /// the older `seq_no`.
    pub(in crate::hnsw::graph) fn edit_links<F>(&mut self, node: SerialId, tick: Tick, f: F)
    where
        F: FnOnce(u64, &mut Vec<SerialId>),
    {
        let Some(nbhd) = self.links.get_mut(&node) else {
            return;
        };
        // Key the set-hash on (node, seq_no) so the neighborhood's freshness
        // certificate is part of the consensus checksum. Remove under the OLD
        // seq_no, re-add under the new one.
        self.set_hash
            .remove_unordered_set((node, nbhd.seq_no), nbhd.neighbors.iter());
        f(nbhd.seq_no, &mut nbhd.neighbors);
        nbhd.neighbors.sort_unstable();
        nbhd.neighbors.dedup();
        nbhd.seq_no = tick.value();
        self.set_hash
            .add_unordered_set((node, nbhd.seq_no), nbhd.neighbors.iter());
    }

    /// Remove a node from the graph and clean up all backlinks from its neighbors.
    pub fn remove_node(&mut self, id: SerialId) {
        if let Some(nbhd) = self.links.remove(&id) {
            self.set_hash
                .remove_unordered_set((id, nbhd.seq_no), nbhd.neighbors.iter());

            // Remove the node from all neighbors' neighborhoods (bidirectional cleanup).
            // note that if this node did compaction then some old neighbors could still have links
            // to this deleted node. that is ok. it is also ok if the following code block is deleted.
            // this is just an opportunistic low-cost cleanup.
            for neighbor in nbhd.neighbors {
                if let Some(neighbor_nbhd) = self.links.get_mut(&neighbor) {
                    // seq_no is unchanged here (only the link list shrinks), so
                    // remove and re-add under the same (neighbor, seq_no) key.
                    self.set_hash.remove_unordered_set(
                        (neighbor, neighbor_nbhd.seq_no),
                        neighbor_nbhd.neighbors.iter(),
                    );
                    neighbor_nbhd.neighbors.retain(|x| *x != id);
                    self.set_hash.add_unordered_set(
                        (neighbor, neighbor_nbhd.seq_no),
                        neighbor_nbhd.neighbors.iter(),
                    );
                }
            }
        }
    }

    /// Replace `from`'s entire neighbor list in one shot, stamping `seq_no`
    /// directly. This is the **bulk-load / trusted-construction** primitive
    /// (deserialization, checkpoint hashing, idealized graphs); it does *not*
    /// filter against the content clock because it sets a full, trusted list
    /// rather than incrementally mutating a live neighborhood. Incremental
    /// growth/shrink of a live neighborhood goes through [`Layer::edit_links`].
    pub fn set_links(&mut self, from: SerialId, links: Vec<SerialId>, seq_no: u64) {
        use std::collections::hash_map::Entry;
        match self.links.entry(from) {
            Entry::Occupied(mut e) => {
                let key = *e.key();
                let old_seq = e.get().seq_no;
                self.set_hash
                    .remove_unordered_set((key, old_seq), e.get().neighbors.iter());
                let existing = e.get_mut();
                existing.neighbors.clear();
                existing.neighbors.extend(links);
                existing.seq_no = seq_no;
                self.set_hash
                    .add_unordered_set((key, seq_no), existing.neighbors.iter());
            }
            Entry::Vacant(e) => {
                self.set_hash
                    .add_unordered_set((*e.key(), seq_no), links.iter());
                e.insert(Neighborhood {
                    neighbors: links,
                    seq_no,
                });
            }
        }
    }

    pub fn get_links_map(&self) -> &HashMap<SerialId, Neighborhood> {
        &self.links
    }

    fn from_knn_results(results: Vec<KNNResult<SerialId>>, n: usize) -> Self {
        let mut ret = Layer::new();
        for KNNResult { node, neighbors } in results.into_iter().take(n) {
            ret.set_links(node, neighbors, 0);
        }
        ret
    }

    /// Constructs a Layer from `(serial_id, K::Vector)` pairs by brute-force
    /// top-k KNN using the supplied [`IdealKnn`] implementation.
    pub fn ideal_from_data<K: IdealKnn>(
        data: Vec<(SerialId, K::Vector)>,
        k: usize,
        knn: K,
    ) -> Self {
        let (serial_ids, vectors): (Vec<SerialId>, Vec<K::Vector>) = data.into_iter().unzip();
        let n = vectors.len();
        if n == 0 {
            return Layer::new();
        }
        let k = k.min(n - 1);

        let mut engine = NaiveKNN::<K>::init(knn, vectors, k, 1);
        let results = engine
            .compute_chunk(n)
            .into_iter()
            // remap from engine 1-based indices to original serial ids
            .map(|result| result.map(|i| serial_ids[(i as usize) - 1]))
            .collect::<Vec<_>>();

        Layer::from_knn_results(results, n)
    }

    /// Layer constructor for iris codes — kept for backward compatibility;
    /// delegates to [`Layer::ideal_from_data`].
    pub fn ideal_from_irises(
        iris_data: Vec<(SerialId, IrisCode)>,
        k: usize,
        echoice: EngineChoice,
    ) -> Self {
        use crate::hawkers::{
            aby3::aby3_store::{FhdOps, NhdOps},
            ideal_knn_engines::IrisKnn,
        };
        let mode = echoice.distance_mode();
        match echoice {
            EngineChoice::NaiveFHD | EngineChoice::NaiveMinFHD => {
                Layer::ideal_from_data::<IrisKnn<FhdOps>>(iris_data, k, IrisKnn::new(mode))
            }
            EngineChoice::NaiveNHD | EngineChoice::NaiveMinNHD => {
                Layer::ideal_from_data::<IrisKnn<NhdOps>>(iris_data, k, IrisKnn::new(mode))
            }
        }
    }

    /// Layer constructor for deep-ID Int4 vectors — kept for backward
    /// compatibility; delegates to [`Layer::ideal_from_data`].
    pub fn ideal_from_int4_vectors(
        data: Vec<(
            SerialId,
            crate::hawkers::plaintext_deep_id_store::Int4Vector,
        )>,
        k: usize,
        echoice: crate::hawkers::ideal_knn_engines::EngineChoiceInt4,
    ) -> Self {
        use crate::hawkers::ideal_knn_engines::{EngineChoiceInt4, Int4DotKnn};
        match echoice {
            EngineChoiceInt4::NaiveInt4Dot => {
                Layer::ideal_from_data::<Int4DotKnn>(data, k, Int4DotKnn)
            }
        }
    }
}

/// Convert a `GraphMem` data structure via a direct mapping of serial ids,
/// leaving the edge sets associated with the mapped vertices unchanged.
///
/// This could be useful for cases where the representation of the graph
/// vertices or distances is changed, but not the underlying values. For
/// example:
/// - serial ids are re-mapped to remove blank entries left by deletions
pub fn migrate<VecMap>(graph: GraphMem, vector_map: VecMap) -> GraphMem
where
    VecMap: Fn(SerialId) -> SerialId + Copy,
{
    let last_update_seq_no = graph.last_update_seq_no;
    let new_entry_points = graph
        .entry_points
        .iter()
        .map(|ep| EntryPoint {
            point: vector_map(ep.point),
            layer: ep.layer,
        })
        .collect();

    let new_layers: Vec<_> = graph
        .layers
        .into_iter()
        .map(|v| {
            let mut layer = Layer::new();
            for (from, nbhd) in v.links.into_iter() {
                layer.set_links(
                    vector_map(from),
                    nbhd.neighbors.into_iter().map(vector_map).collect(),
                    nbhd.seq_no,
                );
            }
            layer
        })
        .collect();

    let new_node_last_update_seq_no = graph
        .node_init_seq_no
        .into_iter()
        .map(|(id, seq)| (vector_map(id), seq))
        .collect();

    GraphMem::from_parts(
        new_entry_points,
        new_layers,
        last_update_seq_no,
        new_node_last_update_seq_no,
    )
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use crate::{
        hawkers::{aby3::aby3_store::FhdOps, plaintext_store::PlaintextStore},
        hnsw::{
            graph::layered_graph::migrate, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
            VectorStore,
        },
    };
    use aes_prng::AesRng;
    use eyre::Result;
    use iris_mpc_common::{iris_db::db::IrisDB, SerialId, VectorId};

    use rand::{RngCore, SeedableRng};

    /// The consensus checksum must change when either clock changes, even with
    /// identical neighbor sets and entry points — otherwise parties that skip
    /// different edges via `is_active` could agree on the checksum.
    #[test]
    fn checksum_folds_content_and_neighborhood_clocks() {
        use super::{EntryPoint, Layer};
        // node1 neighborhood seq = `nbhd_seq`; node1 content clock = `content`.
        let build = |nbhd_seq: u64, content: u64| -> GraphMem {
            let mut layer = Layer::new();
            layer.set_links(1u32, vec![2u32], nbhd_seq);
            layer.set_links(2u32, vec![], 0);
            let node_init = HashMap::from([(1u32, content), (2u32, 0u64)]);
            GraphMem::from_parts(
                vec![EntryPoint { point: 1, layer: 0 }],
                vec![layer],
                0,
                node_init,
            )
        };

        let base = build(0, 0);
        assert_eq!(base.checksum(), build(0, 0).checksum(), "deterministic");
        assert_ne!(
            base.checksum(),
            build(5, 0).checksum(),
            "per-neighborhood seq_no must be folded into the checksum"
        );
        assert_ne!(
            base.checksum(),
            build(0, 7).checksum(),
            "node_init_seq_no (content clock) must be folded into the checksum"
        );
    }

    #[allow(dead_code)]
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
        type DistanceRef = u32; // Eager distance representation as fraction.

        async fn vectors_as_queries(
            &mut self,
            vectors: Vec<VectorId>,
        ) -> Result<Vec<Self::QueryRef>> {
            Ok(vectors
                .into_iter()
                .map(|v| v.serial_id() as usize)
                .collect())
        }

        async fn eval_distance(
            &mut self,
            query: &Self::QueryRef,
            vector: &VectorId,
        ) -> Result<Self::DistanceRef> {
            // Hamming distance
            let vector_1_idx = vector.serial_id() as usize;
            let vector_0 = self.points[query].data;
            let vector_1 = self.points[&vector_1_idx].data;
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

        async fn only_valid_entry_points(
            &mut self,
            entry_points: Vec<(VectorId, usize)>,
        ) -> Vec<(VectorId, usize)> {
            entry_points
        }

        fn serial_to_vector_id(&self, serial_id: SerialId) -> VectorId {
            VectorId::from_serial_id(serial_id)
        }
    }

    impl VectorStoreMut for TestStore {
        async fn insert(&mut self, query: &Self::QueryRef) -> VectorId {
            // The query is now accepted in the store. It keeps the same ID.
            self.points.get_mut(query).unwrap().is_persistent = true;
            VectorId::from_serial_id(*query as u32)
        }

        async fn insert_at(
            &mut self,
            _vector_ref: &VectorId,
            _query: &Self::QueryRef,
        ) -> Result<VectorId> {
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
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
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

        // serial_id is 1-based, so serial_id -> (serial_id - 1) * 2 + 1
        let different_graph_store: GraphMem =
            migrate(graph_store.clone(), |v: SerialId| (v - 1) * 2 + 1);
        assert_ne!(graph_store, different_graph_store);

        Ok(())
    }

    #[test]
    fn test_layer_deterministic_serialize_order() {
        let mut layer_a = super::Layer::new();
        let mut layer_b = super::Layer::new();

        layer_a.set_links(1, vec![2, 3], 0);
        layer_a.set_links(4, vec![5], 0);

        layer_b.set_links(4, vec![5], 0);
        layer_b.set_links(1, vec![2, 3], 0);

        let bytes_a = bincode::serialize(&layer_a).expect("layer_a serialize");
        let bytes_b = bincode::serialize(&layer_b).expect("layer_b serialize");

        assert_eq!(bytes_a, bytes_b);
    }

    #[tokio::test]
    async fn test_from_another() -> Result<()> {
        let mut vector_store = PlaintextStore::<FhdOps>::new();
        let mut graph_store = GraphMem::new();
        let mut searcher = HnswSearcher::new_with_test_parameters();
        // Bump layer density so enough nodes roll onto the entry-point layer
        // (max_graph_layer + 1) for the entry-point migration checks below.
        searcher.layer_distribution =
            crate::hnsw::searcher::LayerDistribution::new_geometric_from_M(2);
        let mut rng = AesRng::seed_from_u64(0_u64);

        let mut point_ids_map: HashMap<SerialId, SerialId> = HashMap::new();

        for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
            let query = Arc::new(raw_query);
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
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

            point_ids_map.insert(inserted.serial_id(), rng.next_u32());
        }

        let new_graph_store: GraphMem = migrate(graph_store.clone(), |v| point_ids_map[&v]);

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
            for (point_id, nbhd) in links.iter() {
                let new_point_id = point_ids_map[point_id];
                let new_nbhd = &new_links[&new_point_id];
                for (neighbor_id, new_neighbor_id) in
                    nbhd.neighbors.iter().zip(new_nbhd.neighbors.iter())
                {
                    assert_eq!(point_ids_map[neighbor_id], *new_neighbor_id);
                }
            }
        }

        Ok(())
    }

    use crate::hnsw::graph::mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint};

    #[test]
    fn add_edges_outgoing_writes_only_to_id_list() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let c = VectorId::from_serial_id(3);
        // Seed: a, b, c all exist at layer 0 with no edges.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 1 },
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddNode {
                        id: c,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                ],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b, c],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors,
            vec![2u32, 3u32]
        );
        assert_eq!(
            graph.layers[0].get_links(&2).unwrap().neighbors,
            vec![] as Vec<SerialId>
        );
        assert_eq!(
            graph.layers[0].get_links(&3).unwrap().neighbors,
            vec![] as Vec<SerialId>
        );
    }

    #[test]
    fn add_edges_incoming_writes_only_to_target_lists() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let c = VectorId::from_serial_id(3);
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 1 },
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddNode {
                        id: c,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                ],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b, c],
                    edge_type: EdgeType::Neighbors,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors,
            vec![] as Vec<SerialId>
        );
        assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
        assert_eq!(graph.layers[0].get_links(&3).unwrap().neighbors, vec![1u32]);
    }

    /// Filter-on-bump: an asymmetric stale edge `a -> b` (no `b -> a` back-edge,
    /// as arises after compaction) is dropped the next time `a`'s neighborhood
    /// is touched once `b`'s content clock has advanced via reauth. This is the
    /// v5 replacement for main's per-edge version-skip.
    #[test]
    fn stale_edge_dropped_on_neighborhood_touch_after_reauth() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let d = VectorId::from_serial_id(3);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: insert a and b.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // seq 2: asymmetric forward edge a -> b only; b's list stays empty.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );

        // seq 3: reauth b. RemoveNode(b) — b's list is empty, so its
        // opportunistic bidirectional cleanup never touches a's list and a -> b
        // survives — then AddNode(b) advances content[b] to 3.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
            })
            .unwrap();
        // a's neighborhood (last touched at seq 2) hasn't been revisited, so the
        // stale edge is still present — detection happens on next touch.
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );

        // seq 4/5: touch a's neighborhood with a fresh edge a -> d. The filter
        // runs against a's prior seq (2) and drops a -> b (content[b] = 3 > 2)
        // before appending d.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 4,
                ops: vec![node(d)],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 5,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![d],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![3u32]
        );
    }

    /// Read-path skip: `get_active_links` omits a content-stale neighbor even
    /// when the neighborhood was never touched after the reauth (so the physical
    /// edge is still present). This is what makes search match main's
    /// version-skip for stale edges that filter-on-bump hasn't yet cleaned.
    #[test]
    fn get_active_links_skips_content_stale_neighbor() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // Asymmetric a -> b; a's neighborhood certified at seq 2.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(graph.get_active_links(&1, 0), vec![2u32]);

        // Reauth b (content[b] -> 3) without touching a.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
            })
            .unwrap();

        // Physical edge survives (a untouched), but read-path skips it.
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );
        assert_eq!(graph.get_active_links(&1, 0), Vec::<SerialId>::new());
    }

    /// Read-path liveness: `get_active_links` drops a dangling edge to a removed
    /// node even though the physical edge survives `RemoveNode`'s opportunistic
    /// cleanup. The removed node has no content-clock entry, so `is_active`
    /// treats it as dead.
    #[test]
    fn get_active_links_drops_edge_to_removed_node() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // Asymmetric a -> b so b's removal can't clean a's back-edge.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        // Remove b (no re-insert): content[b] is dropped.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                ops: vec![MutationOp::RemoveNode { id: b }],
            })
            .unwrap();

        // Physical edge a -> b survives, but read-path skips the dead node.
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );
        assert_eq!(graph.get_active_links(&1, 0), Vec::<SerialId>::new());
    }

    #[test]
    fn add_edges_bidirectional_writes_both_sides() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let c = VectorId::from_serial_id(3);
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 1 },
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddNode {
                        id: c,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                ],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b, c],
                    edge_type: EdgeType::All,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors,
            vec![2u32, 3u32]
        );
        assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
        assert_eq!(graph.layers[0].get_links(&3).unwrap().neighbors, vec![1u32]);
    }

    #[test]
    fn remove_edges_outgoing_only_modifies_id_list() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let c = VectorId::from_serial_id(3);
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 1 },
                    },
                    MutationOp::AddEdges {
                        base: a,
                        layer: 0,
                        neighbors: vec![b, c],
                        edge_type: EdgeType::Base,
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddEdges {
                        base: b,
                        layer: 0,
                        neighbors: vec![a],
                        edge_type: EdgeType::Base,
                    },
                    MutationOp::AddNode {
                        id: c,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddEdges {
                        base: c,
                        layer: 0,
                        neighbors: vec![a],
                        edge_type: EdgeType::Base,
                    },
                ],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                ops: vec![MutationOp::RemoveEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(graph.layers[0].get_links(&1).unwrap().neighbors, vec![3u32]);
        // Bidirectional cleanup is not implied — b's list still contains a.
        assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
    }

    #[test]
    fn two_phase_apply_edges_before_node_in_vec_still_works() {
        // Pass 1 should apply AddNode before pass 2 applies AddEdges, regardless
        // of their order in the input Vec.
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                ops: vec![
                    // Listed first: an edge op that references a node not yet created.
                    MutationOp::AddEdges {
                        base: a,
                        layer: 0,
                        neighbors: vec![b],
                        edge_type: EdgeType::Base,
                    },
                    // Listed second: the node creation.
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 0 },
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                ],
            })
            .unwrap();
        // Pass-1 created the nodes, then pass-2 applied the edge — so a should
        // now have b in its outgoing list.
        assert_eq!(graph.layers[0].get_links(&1).unwrap().neighbors, vec![2u32]);
    }

    #[test]
    fn next_seq_no_is_one_past_last_and_does_not_mutate() {
        use crate::hnsw::GraphMem;
        let mut graph = GraphMem::new();
        assert_eq!(graph.last_update_seq_no, 0);
        assert_eq!(graph.next_sequence_number(), 1);
        assert_eq!(graph.next_sequence_number(), 1, "peek must not mutate");
        graph.last_update_seq_no = 42;
        assert_eq!(graph.next_sequence_number(), 43);
        assert_eq!(graph.last_update_seq_no, 42, "peek must not mutate");
    }

    #[test]
    fn insert_apply_advances_last_update_seq_no_on_success() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let mutation = GraphMutation {
            seq_no: 1,
            ops: vec![MutationOp::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        };
        graph
            .insert_apply(&mutation)
            .expect("strict-increase should hold");
        assert_eq!(graph.last_update_seq_no, 1);
    }

    #[test]
    fn insert_apply_rejects_seq_no_equal_to_last_update_seq_no() {
        let mut graph = GraphMem::new();
        graph.last_update_seq_no = 5;
        let mutation = GraphMutation {
            seq_no: 5,
            ops: vec![MutationOp::AddNode {
                id: VectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        };
        let res = graph.insert_apply(&mutation);
        assert!(res.is_err(), "equal seq_no must be rejected");
        assert_eq!(
            graph.last_update_seq_no, 5,
            "state must be unchanged on Err"
        );
        assert_eq!(graph.layers.len(), 0, "no ops should have been applied");
    }

    #[test]
    fn insert_apply_rejects_seq_no_below_last_update_seq_no() {
        let mut graph = GraphMem::new();
        graph.last_update_seq_no = 10;
        let mutation = GraphMutation {
            seq_no: 9,
            ops: vec![MutationOp::AddNode {
                id: VectorId::from_serial_id(1),
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        };
        let res = graph.insert_apply(&mutation);
        assert!(res.is_err());
        assert_eq!(graph.last_update_seq_no, 10);
    }

    #[test]
    fn insert_apply_all_short_circuits_on_first_violation() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let mutations = vec![
            GraphMutation {
                seq_no: 1,
                ops: vec![MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 0 },
                }],
            },
            // Equal seq_no — should fail.
            GraphMutation {
                seq_no: 1,
                ops: vec![MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                }],
            },
        ];
        let res = graph.insert_apply_all(&mutations);
        assert!(res.is_err(), "second mutation must be rejected");
        assert_eq!(
            graph.last_update_seq_no, 1,
            "first applied; last_update_seq_no at 1"
        );
        // First mutation's AddNode took effect, second did not.
        assert!(graph.layers[0].get_links(&1).is_some());
        assert!(graph.layers[0].get_links(&2).is_none());
    }
}

#[cfg(test)]
mod int4_layer_tests {
    use super::*;
    use crate::hawkers::{
        ideal_knn_engines::EngineChoiceInt4, plaintext_deep_id_store::Int4Vector,
    };
    use aes_prng::AesRng;
    use rand::SeedableRng;

    #[test]
    fn layer_ideal_from_int4_vectors_matches_brute_force() {
        let mut rng = AesRng::seed_from_u64(0xC0FFEE);
        let n = 12_usize;
        let k = 3_usize;
        let vectors: Vec<Int4Vector> = (0..n).map(|_| Int4Vector::random(&mut rng)).collect();

        let data: Vec<(SerialId, Int4Vector)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| ((i + 1) as u32, v.clone()))
            .collect();

        let layer = Layer::ideal_from_int4_vectors(data, k, EngineChoiceInt4::NaiveInt4Dot);

        #[allow(
            clippy::iter_over_hash_type,
            reason = "Iteration is over a parallel data structure, compared entry by entry."
        )]
        for (key, nbhd) in layer.get_links_map() {
            let me_idx = *key as usize - 1;
            let me = &vectors[me_idx];
            let mut dists: Vec<(SerialId, i32)> = (0..n)
                .filter(|j| *j != me_idx)
                .map(|j| ((j + 1) as u32, me.dot(&vectors[j])))
                .collect();
            dists.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let expected: Vec<SerialId> = dists.into_iter().take(k).map(|(j, _)| j).collect();
            assert_eq!(&nbhd.neighbors, &expected, "key {key}");
        }
    }
}
