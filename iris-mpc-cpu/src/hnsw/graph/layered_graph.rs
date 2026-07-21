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
use iris_mpc_common::{iris_db::iris::IrisCode, SerialId, VectorId, VersionId};
use itertools::{izip, Itertools};
use serde::{
    ser::{SerializeMap, SerializeStruct, Serializer},
    Deserialize, Serialize,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Display,
    iter::once,
    path::PathBuf,
    sync::Arc,
};
use tokio::sync::RwLock;
use tracing::warn;

/// Sequence number to stamp on a neighborhood edit. Constructible only within
/// the graph module, so `seq_no` stamps cannot be minted out of band.
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

/// One node's neighbor list in one layer, plus the seq_no of the mutation that
/// last modified it.
///
/// Fields are private: the "every edge valid as of `seq_no`" invariant is
/// maintained solely by [`Layer`]'s mutators.
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

    /// Seq_no of the mutation that last modified this neighborhood; every edge
    /// in `neighbors` is valid as of this tick.
    pub fn seq_no(&self) -> u64 {
        self.seq_no
    }
}

/// Content-clock entry of one live node: the seq_no of its last (re-)insertion
/// and the iris version it was inserted at. Makes the graph self-contained for
/// resolving in-graph serials to current `VectorId`s
/// ([`GraphMem::vector_id_of`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInit {
    /// Seq_no of the node's last `AddNode`. Gates edge liveness via [`is_active`].
    pub seq_no: u64,
    /// Iris version carried by that `AddNode`'s `VectorId`.
    pub version: VersionId,
}

impl NodeInit {
    /// Whether this node's content is unchanged since `old_seq` — the sole
    /// definition of the staleness comparison; see [`is_active`].
    fn active_at(self, old_seq: u64) -> bool {
        self.seq_no <= old_seq
    }
}

/// Whether edge `A -> z` is valid for a neighborhood last certified at
/// `old_seq`: `z` is live (present in the content clock) and its content has
/// not advanced past `old_seq`. Absent means removed; `> old_seq` means
/// reauthed since the edge was certified — either way the edge is invalid.
///
/// The lazy realization of "removing or re-inserting a node invalidates every
/// edge incident to it", applied at three anchors: edits filter against the
/// neighborhood's old certificate (`GraphMem::edit_neighborhood`), reads
/// against its current one ([`GraphMem::get_active_links`]), and every apply
/// resolves the record's references against the record's `as_of`
/// ([`GraphMem::resolve_ops`]).
fn is_active(content: &HashMap<SerialId, NodeInit>, z: SerialId, old_seq: u64) -> bool {
    content.get(&z).is_some_and(|ni| ni.active_at(old_seq))
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

    /// Sequence number of the most recently applied `GraphMutation`; `0` means
    /// none. Advanced on every successful apply.
    pub last_update_seq_no: u64,

    /// Content clock: per live node, the seq_no of its last (re-)insertion and
    /// the iris version it carried. Moved only by the node's own
    /// `AddNode`/`RemoveNode`, never by edge ops. Gates edge liveness
    /// ([`is_active`]) and resolves serials to `VectorId`s
    /// ([`GraphMem::vector_id_of`]).
    pub node_init: HashMap<SerialId, NodeInit>,

    /// Order-agnostic hash of `node_init`, folded into [`GraphMem::checksum`]
    /// so the content clock is part of cross-party consensus. Derived: not
    /// serialized, recomputed by `from_parts`, kept in sync by the mutation
    /// apply. Private so all construction routes through `from_parts`.
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
            node_init: self.node_init.clone(),
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
            node_init: HashMap<SerialId, NodeInit>,
        }
        let d = GraphMemData::deserialize(deserializer)?;
        Ok(GraphMem::from_parts(
            d.entry_points,
            d.layers,
            d.last_update_seq_no,
            d.node_init,
        ))
    }
}

impl GraphMem {
    pub fn new() -> Self {
        GraphMem {
            entry_points: vec![],
            layers: vec![],
            last_update_seq_no: 0,
            node_init: HashMap::new(),
            node_init_hash: SetHash::default(),
        }
    }

    /// Assemble a `GraphMem` from its parts, computing the derived
    /// `node_init_hash`. Sole constructor for callers outside this module.
    pub fn from_parts(
        entry_points: Vec<EntryPoint>,
        layers: Vec<Layer>,
        last_update_seq_no: u64,
        node_init: HashMap<SerialId, NodeInit>,
    ) -> Self {
        let mut node_init_hash = SetHash::default();
        // SetHash folds via commutative wrapping addition, so iteration order is
        // irrelevant to the result.
        #[allow(clippy::iter_over_hash_type)]
        for (&serial, &init) in &node_init {
            node_init_hash.add_unordered(Self::node_init_contribution(serial, init));
        }
        GraphMem {
            entry_points,
            layers,
            last_update_seq_no,
            node_init,
            node_init_hash,
        }
    }

    /// How a content-clock entry folds into `node_init_hash`; shared by the
    /// bulk build (`from_parts`) and the incremental apply so they can't drift.
    fn node_init_contribution(
        serial: SerialId,
        init: NodeInit,
    ) -> (&'static str, SerialId, u64, VersionId) {
        ("node_init", serial, init.seq_no, init.version)
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
        // Seed the content clock at seq 0 / version 0 so `is_active` treats
        // these trusted nodes as live.
        let node_init = layers
            .iter()
            .flat_map(|l| l.links.keys())
            .map(|&v| {
                (
                    v,
                    NodeInit {
                        seq_no: 0,
                        version: 0,
                    },
                )
            })
            .collect();
        let entry_points = entry_points
            .into_iter()
            .map(|ep| EntryPoint {
                point: ep.0,
                layer: ep.1,
            })
            .collect::<Vec<_>>();
        GraphMem::from_parts(entry_points, layers, 0, node_init)
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

    /// Apply a mutation record to the in-memory graph.
    ///
    /// The one apply path: minting ([`Self::apply_new`]) and replay both land
    /// here. Edge references are resolved against the record's `as_of`
    /// ([`Self::resolve_ops`]) and staleness cleanup re-derives from graph
    /// state, so replay reaches the same state the mint produced.
    ///
    /// # Errors
    /// `mutation.seq_no` must be strictly greater than `last_update_seq_no`;
    /// otherwise returns `Err` without touching the graph.
    pub fn insert_apply(&mut self, mutation: &GraphMutation) -> Result<()> {
        self.apply_ops(mutation.seq_no, mutation.as_of, &mutation.ops)
    }

    /// Resolve the record's references at its `as_of` ([`Self::resolve_ops`]),
    /// then two passes: node ops, then edge ops. Every touched neighborhood is
    /// filtered through [`is_active`] before the op's own edit
    /// (filter-on-bump), so a re-stamped freshness certificate never covers an
    /// invalid edge.
    ///
    /// Causal construction: an `AddEdges` reference to a target created in a
    /// *later* mutation is not yet in the content clock, so resolution drops
    /// it. Insert endpoints before wiring edges between them; debug builds
    /// assert it.
    fn apply_ops(&mut self, seq_no: u64, as_of: u64, ops: &[MutationOp]) -> Result<()> {
        if seq_no <= self.last_update_seq_no {
            return Err(eyre::eyre!(
                "GraphMem::apply_ops: mutation seq_no {} is not strictly greater than \
                 last_update_seq_no {}",
                seq_no,
                self.last_update_seq_no,
            ));
        }
        debug_assert!(
            as_of < seq_no,
            "record as_of {as_of} is not before its own seq_no {seq_no}",
        );
        let ops = self.resolve_ops(as_of, ops);

        // One tick for this mutation; both passes stamp neighborhoods with it.
        let tick = Tick::new(seq_no);

        // Pass 1: apply node-level mutations.
        for op in ops.iter() {
            match op {
                MutationOp::RemoveNode { id } => {
                    let sid = id.serial_id();
                    for layer in &mut self.layers {
                        layer.remove_node(sid);
                    }
                    self.entry_points.retain(|ep| ep.point != sid);
                    if let Some(old) = self.node_init.remove(&sid) {
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
                        self.layers[layer_idx].create_node(sid, tick);
                    }
                    let init = NodeInit {
                        seq_no,
                        version: id.version_id(),
                    };
                    if let Some(old) = self.node_init.insert(sid, init) {
                        self.node_init_hash
                            .remove(Self::node_init_contribution(sid, old));
                    }
                    self.node_init_hash
                        .add_unordered(Self::node_init_contribution(sid, init));
                }
                MutationOp::AddEdges { .. } | MutationOp::RemoveEdges { .. } => {}
            }
        }

        // Pass 2: edge-level mutations, every neighborhood touch routed through
        // `Self::edit_neighborhood`. Edge ops never advance the content clock.
        for op in ops.iter() {
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
                    // An edge to a node whose clock is minted later reads as
                    // stale to `is_active` and is silently dropped; endpoints
                    // must exist before edges wire to them.
                    debug_assert!(
                        to_add.iter().all(|z| self.node_init.contains_key(z)),
                        "AddEdges: neighbor absent from content clock (edge added \
                         before its endpoint exists); the edge would be silently dropped"
                    );
                    // Forward half: append `to_add` to base's own list.
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        if self.layers[layer].get_links(base).is_none() {
                            warn!("AddEdges({edge_type:?}): base={base} missing at layer {layer}; skipping outgoing half");
                        } else {
                            self.edit_neighborhood(layer, *base, tick, |nbrs| {
                                nbrs.extend_from_slice(to_add)
                            });
                        }
                    }
                    // Back half: append base into each target's list.
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for target in to_add.iter() {
                            if self.layers[layer].get_links(target).is_none() {
                                warn!("AddEdges({edge_type:?}): target={target} missing at layer {layer} (base={base}); skipping back-edge");
                            } else {
                                self.edit_neighborhood(layer, *target, tick, |nbrs| {
                                    nbrs.push(*base)
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
                        warn!("RemoveEdges: layer {layer} does not exist (base={base}); skipping");
                        continue;
                    }
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        if self.layers[layer].get_links(base).is_none() {
                            warn!("RemoveEdges({edge_type:?}): base={base} missing at layer {layer}; skipping outgoing half");
                        } else {
                            self.edit_neighborhood(layer, *base, tick, |nbrs| {
                                nbrs.retain(|z| !to_remove.contains(z))
                            });
                        }
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for target in to_remove.iter() {
                            if self.layers[layer].get_links(target).is_none() {
                                warn!("RemoveEdges({edge_type:?}): target={target} missing at layer {layer} (base={base}); skipping");
                            } else {
                                self.edit_neighborhood(layer, *target, tick, |nbrs| {
                                    nbrs.retain(|z| *z != *base)
                                });
                            }
                        }
                    }
                }
            }
        }

        self.last_update_seq_no = seq_no;
        Ok(())
    }

    /// Edit `node`'s neighborhood at layer `lc`: edges invalid against the
    /// *old* certificate ([`is_active`]) are dropped before `f` runs, then the
    /// neighborhood is re-stamped at `tick` — an append can't re-certify an
    /// already-invalid sibling. Sole graph-code path to [`Layer::edit_links`].
    fn edit_neighborhood<F>(&mut self, lc: usize, node: SerialId, tick: Tick, f: F)
    where
        F: FnOnce(&mut Vec<SerialId>),
    {
        let content = &self.node_init;
        self.layers[lc].edit_links(node, tick, |old_seq, nbrs| {
            nbrs.retain(|z| is_active(content, *z, old_seq));
            f(nbrs);
            debug_assert!(
                nbrs.iter().all(|z| is_active(content, *z, tick.value())),
                "edit left an invalid edge in a re-stamped neighborhood"
            );
        });
    }

    /// Stamp an [`UnstampedMutation`] with the next sequence number and apply
    /// it, returning the [`GraphMutation`] record carrying ops and `as_of`
    /// verbatim; reference resolution happens inside the apply, identically
    /// here and on replay.
    ///
    /// Sole minter of sequence numbers: assignment and apply are one step, so
    /// `last_update_seq_no` never lags a minted id and no two mutations share
    /// a number. Records that already carry a `seq_no` (WAL/checkpoint replay)
    /// go through `insert_apply` instead.
    pub fn apply_new(&mut self, mutation: UnstampedMutation) -> Result<GraphMutation> {
        let record = GraphMutation {
            seq_no: self.next_sequence_number(),
            as_of: mutation.as_of,
            ops: mutation.ops,
        };
        self.insert_apply(&record)?;
        Ok(record)
    }

    /// Resolve a record's edge references, identified at `as_of`, against the
    /// current graph. A `neighbors` serial is *void* if it is not
    /// [`is_active`] at `as_of` — removed, or reauthed after identification —
    /// unless this record's own `AddNode` creates it; void neighbors are
    /// dropped from their op. A void `RemoveEdges` target is skipped: the
    /// ranked edge is already invalid, and a fresh edge to the target's new
    /// content is not this record's to evict.
    ///
    /// A `base` names the current node with that serial, whatever its
    /// content. An op whose base has no current node (and none minted here)
    /// is dropped whole — edges must not wire to an absent endpoint.
    ///
    /// `AddEdges` additionally drops self-references (`z == base`): in a
    /// minting record they are stale echoes of a search against the node's
    /// replaced content, so the own-`AddNode` carve-out must not re-admit
    /// them. Dropped silently (routine on reauth). `RemoveEdges` keeps
    /// self-references, so removing an existing self-edge stays expressible.
    ///
    /// Node ops pass through untouched. Deterministic in the graph state, so
    /// replay resolves each record exactly as its mint did.
    fn resolve_ops(&self, as_of: u64, ops: &[MutationOp]) -> Vec<MutationOp> {
        let minted: HashSet<SerialId> = ops
            .iter()
            .filter_map(|op| match op {
                MutationOp::AddNode { id, .. } => Some(id.serial_id()),
                _ => None,
            })
            .collect();
        let fresh = |z: &SerialId| minted.contains(z) || is_active(&self.node_init, *z, as_of);
        let exists = |z: &SerialId| minted.contains(z) || self.node_init.contains_key(z);

        let mut dropped = 0usize;
        // The surviving refs of one op, or `None` if the op is void.
        let mut filter_refs = |base: &SerialId, neighbors: &[SerialId], keep_self: bool| {
            if !exists(base) {
                dropped += neighbors.len();
                return None;
            }
            let mut kept = Vec::with_capacity(neighbors.len());
            for z in neighbors {
                if !keep_self && z == base {
                    continue;
                }
                if fresh(z) {
                    kept.push(*z);
                } else {
                    dropped += 1;
                }
            }
            (!kept.is_empty()).then_some(kept)
        };

        let mut resolved = Vec::with_capacity(ops.len());
        for op in ops {
            let resolved_op = match op {
                MutationOp::AddNode { .. } | MutationOp::RemoveNode { .. } => Some(op.clone()),
                MutationOp::AddEdges {
                    base,
                    neighbors,
                    layer,
                    edge_type,
                } => filter_refs(base, neighbors, false).map(|neighbors| MutationOp::AddEdges {
                    base: *base,
                    neighbors,
                    layer: *layer,
                    edge_type: edge_type.clone(),
                }),
                MutationOp::RemoveEdges {
                    base,
                    neighbors,
                    layer,
                    edge_type,
                } => filter_refs(base, neighbors, true).map(|neighbors| MutationOp::RemoveEdges {
                    base: *base,
                    neighbors,
                    layer: *layer,
                    edge_type: edge_type.clone(),
                }),
            };
            resolved.extend(resolved_op);
        }
        if dropped > 0 {
            metrics::counter!("graph_resolution_dropped_refs").increment(dropped as u64);
            warn!(
                "resolution dropped {dropped} edge ref(s) identified at seq {as_of} \
                 (graph at {})",
                self.last_update_seq_no,
            );
        }
        resolved
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

    /// The stored neighbor list of `base` at layer `lc`, verbatim — no
    /// staleness filtering. For maintenance paths that reason about physical
    /// edges; traversal must use [`Self::get_active_links`]. Empty if
    /// `base`/`lc` absent.
    pub fn get_raw_links(&self, base: &SerialId, lc: usize) -> &[SerialId] {
        self.layers
            .get(lc)
            .and_then(|layer| layer.get_links(base))
            .map(|n| n.neighbors.as_slice())
            .unwrap_or(&[])
    }

    /// Current `VectorId` of an in-graph node, from the content clock. `None`
    /// means not live — callers must not fabricate a version for it.
    pub fn vector_id_of(&self, serial: SerialId) -> Option<VectorId> {
        self.node_init
            .get(&serial)
            .map(|ni| VectorId::new(serial, ni.version))
    }

    /// Neighbors of `base` at layer `lc` valid for traversal, resolved to
    /// current `VectorId`s: `z` is kept iff [`is_active`] against this
    /// neighborhood's certified `seq_no`, so reauthed/removed edges are
    /// skipped even in neighborhoods no write has physically cleaned yet.
    /// Empty if `base`/`lc` absent.
    pub fn get_active_links(&self, base: &SerialId, lc: usize) -> Vec<VectorId> {
        let Some(nbhd) = self.layers.get(lc).and_then(|layer| layer.get_links(base)) else {
            return Vec::new();
        };
        let old_seq = nbhd.seq_no;
        nbhd.neighbors
            .iter()
            .filter_map(|&z| {
                self.node_init
                    .get(&z)
                    .filter(|ni| ni.active_at(old_seq))
                    .map(|ni| VectorId::new(z, ni.version))
            })
            .collect()
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn checksum(&self) -> u64 {
        let mut set_hash = SetHash::default();
        set_hash.add_unordered_set("entry_points", self.entry_points.iter());
        for (lc, layer) in self.layers.iter().enumerate() {
            set_hash.add_unordered((lc as u64, layer.set_hash.checksum()));
        }
        // Fold the content clock: parties must agree on which edges `is_active`
        // would skip.
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
        let sorted_node_init: BTreeMap<_, _> = self.node_init.iter().collect();
        state.serialize_field("node_init", &sorted_node_init)?;
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

    /// Checksum of this layer's link map, agnostic to `HashMap` iteration
    /// order (neighborhoods fold commutatively).
    pub fn checksum(&self) -> u64 {
        self.set_hash.checksum()
    }

    /// Set-hash contribution of one neighborhood, keyed on `(node, seq_no)`
    /// over the neighbor list. The list MUST be sorted+deduped: the hash is
    /// order-sensitive, and cross-party consensus needs one canonical order.
    /// `edit_links` and `set_links_trusted` maintain that invariant.
    fn neighborhood_contribution(node: SerialId, seq_no: u64, neighbors: &[SerialId]) -> u64 {
        debug_assert!(
            neighbors.is_sorted(),
            "neighborhood must be sorted before hashing (consensus invariant)"
        );
        SetHash::hash((node, seq_no, neighbors))
    }

    pub fn insert_node(&mut self, id: SerialId, neighbors: Vec<SerialId>, seq_no: u64) {
        self.set_links_trusted(id, neighbors, seq_no);
    }

    /// Edit `node`'s existing neighbor list and re-stamp it at `tick`. No-op
    /// if `node` is absent. Representation primitive only — graph code routes
    /// through `GraphMem::edit_neighborhood`, which owns the staleness filter.
    pub(in crate::hnsw::graph) fn edit_links<F>(&mut self, node: SerialId, tick: Tick, f: F)
    where
        F: FnOnce(u64, &mut Vec<SerialId>),
    {
        let Some(nbhd) = self.links.get_mut(&node) else {
            return;
        };
        // The set-hash keys on (node, seq_no), so a re-stamp must rebalance it.
        self.set_hash.remove_hash(Self::neighborhood_contribution(
            node,
            nbhd.seq_no,
            &nbhd.neighbors,
        ));
        f(nbhd.seq_no, &mut nbhd.neighbors);
        nbhd.neighbors.sort_unstable();
        nbhd.neighbors.dedup();
        nbhd.seq_no = tick.value();
        self.set_hash.add_hash(Self::neighborhood_contribution(
            node,
            nbhd.seq_no,
            &nbhd.neighbors,
        ));
    }

    /// Create `node`'s empty neighborhood stamped at `tick`; resets the list
    /// if `node` already exists. Insertion counterpart to [`Self::edit_links`].
    pub(in crate::hnsw::graph) fn create_node(&mut self, node: SerialId, tick: Tick) {
        self.set_links_trusted(node, Vec::new(), tick.value());
    }

    /// Remove `id`'s own neighborhood. Backlinks (`other -> id`) are
    /// intentionally left in place: [`is_active`] masks them at read time, and
    /// the next touch of each holder drops them physically. Keeps node removal
    /// free of implicit edits to other nodes' neighborhoods.
    pub fn remove_node(&mut self, id: SerialId) {
        if let Some(nbhd) = self.links.remove(&id) {
            self.set_hash.remove_hash(Self::neighborhood_contribution(
                id,
                nbhd.seq_no,
                &nbhd.neighbors,
            ));
        }
    }

    /// Trusted bulk-load / construction only. Writes `from`'s full neighbor
    /// list at a caller-supplied raw `seq_no` — no [`Tick`] guard, no
    /// staleness filter — canonicalizing it (sort+dedup). Never call on the
    /// live mutation path; use [`Layer::create_node`] and [`Layer::edit_links`]
    /// there.
    pub fn set_links_trusted(&mut self, from: SerialId, mut links: Vec<SerialId>, seq_no: u64) {
        use std::collections::hash_map::Entry;
        // Canonical order for `neighborhood_contribution`.
        links.sort_unstable();
        links.dedup();
        match self.links.entry(from) {
            Entry::Occupied(mut e) => {
                let key = *e.key();
                let old_seq = e.get().seq_no;
                self.set_hash.remove_hash(Self::neighborhood_contribution(
                    key,
                    old_seq,
                    &e.get().neighbors,
                ));
                let existing = e.get_mut();
                existing.neighbors = links;
                existing.seq_no = seq_no;
                self.set_hash.add_hash(Self::neighborhood_contribution(
                    key,
                    seq_no,
                    &existing.neighbors,
                ));
            }
            Entry::Vacant(e) => {
                self.set_hash
                    .add_hash(Self::neighborhood_contribution(*e.key(), seq_no, &links));
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
            ret.set_links_trusted(node, neighbors, 0);
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
                layer.set_links_trusted(
                    vector_map(from),
                    nbhd.neighbors.into_iter().map(vector_map).collect(),
                    nbhd.seq_no,
                );
            }
            layer
        })
        .collect();

    let new_node_last_update_seq_no = graph
        .node_init
        .into_iter()
        .map(|(id, init)| (vector_map(id), init))
        .collect();

    GraphMem::from_parts(
        new_entry_points,
        new_layers,
        last_update_seq_no,
        new_node_last_update_seq_no,
    )
}

#[cfg(test)]
mod tests;

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
            // Select the top-k by distance, then compare as a set: neighbor lists
            // are stored serial-sorted (canonical for the set-hash), so distance
            // order is not preserved — only membership of the correct k matters.
            let mut expected: Vec<SerialId> = dists.into_iter().take(k).map(|(j, _)| j).collect();
            expected.sort_unstable();
            assert_eq!(&nbhd.neighbors, &expected, "key {key}");
        }
    }
}
