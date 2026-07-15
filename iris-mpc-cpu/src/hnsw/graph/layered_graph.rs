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
        use super::{EntryPoint, Layer, NodeInit};
        // node1 neighborhood seq = `nbhd_seq`; node1 content clock = `content`.
        let build = |nbhd_seq: u64, content: u64, version: i16| -> GraphMem {
            let mut layer = Layer::new();
            layer.set_links_trusted(1u32, vec![2u32], nbhd_seq);
            layer.set_links_trusted(2u32, vec![], 0);
            let node_init = HashMap::from([
                (
                    1u32,
                    NodeInit {
                        seq_no: content,
                        version,
                    },
                ),
                (
                    2u32,
                    NodeInit {
                        seq_no: 0,
                        version: 0,
                    },
                ),
            ]);
            GraphMem::from_parts(
                vec![EntryPoint { point: 1, layer: 0 }],
                vec![layer],
                0,
                node_init,
            )
        };

        let base = build(0, 0, 0);
        assert_eq!(base.checksum(), build(0, 0, 0).checksum(), "deterministic");
        assert_ne!(
            base.checksum(),
            build(5, 0, 0).checksum(),
            "per-neighborhood seq_no must be folded into the checksum"
        );
        assert_ne!(
            base.checksum(),
            build(0, 7, 0).checksum(),
            "node_init (content clock) must be folded into the checksum"
        );
        assert_ne!(
            base.checksum(),
            build(0, 0, 3).checksum(),
            "node version must be folded into the checksum"
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

    #[test]
    fn test_layer_deterministic_serialize_order() {
        let mut layer_a = super::Layer::new();
        let mut layer_b = super::Layer::new();

        layer_a.set_links_trusted(1, vec![2, 3], 0);
        layer_a.set_links_trusted(4, vec![5], 0);

        layer_b.set_links_trusted(4, vec![5], 0);
        layer_b.set_links_trusted(1, vec![2, 3], 0);

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
            let (neighbors, update_ep, as_of) = searcher
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
                    as_of,
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
                // Neighbor lists are stored serial-sorted, and the id remap does
                // not preserve that order, so compare the remapped sets.
                let expected: std::collections::HashSet<SerialId> =
                    nbhd.neighbors.iter().map(|n| point_ids_map[n]).collect();
                let got: std::collections::HashSet<SerialId> =
                    new_nbhd.neighbors.iter().copied().collect();
                assert_eq!(expected, got, "neighbors of {point_id} -> {new_point_id}");
            }
        }

        Ok(())
    }

    use crate::hnsw::graph::mutation::{
        EdgeType, GraphMutation, MutationOp, UnstampedMutation, UpdateEntryPoint,
    };

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
                as_of: 0,
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
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2, 3],
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
                as_of: 0,
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
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2, 3],
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

    /// Filter-on-bump: an asymmetric stale edge `a -> b` (no `b -> a`
    /// back-edge, as arises after compaction) is dropped the next time `a`'s
    /// neighborhood is touched once `b`'s content clock has advanced via
    /// reauth.
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
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // seq 2: asymmetric forward edge a -> b only; b's list stays empty.
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );

        // seq 3: reauth b. RemoveNode(b) does no backlink cleanup, so a -> b
        // survives — then AddNode(b) advances content[b] to 3.
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
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
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![node(d)],
            })
            .unwrap();
        let touch = MutationOp::AddEdges {
            base: 1,
            layer: 0,
            neighbors: vec![3],
            edge_type: EdgeType::Base,
        };
        let minted = graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![touch.clone()],
            })
            .unwrap();
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![3u32]
        );

        // The drop is not reflected in the op list; it re-derives on replay.
        assert_eq!(minted.ops, vec![touch]);
    }

    /// Resolution: an `AddEdges` record identified before its target's
    /// reauth must not re-point at — and certify — the target's new content.
    /// The record carries the intent verbatim; the reference is void at every
    /// apply, so replay reproduces the same graph.
    #[test]
    fn resolution_drops_edge_ref_to_target_reauthed_after_identification() {
        let mut graph = GraphMem::new();
        let mut wal: Vec<GraphMutation> = Vec::new();
        let x0 = VectorId::new(1, 0);
        let x1 = VectorId::new(1, 1);
        let y = VectorId::from_serial_id(2);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: X enters the graph.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![node(x0)],
                })
                .unwrap(),
        );
        // Y's insert search identified X here.
        let as_of = graph.last_update_seq_no;
        // seq 2: X reauths — its content is no longer what the search saw.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![MutationOp::RemoveNode { id: x1 }, node(x1)],
                })
                .unwrap(),
        );
        // seq 3: Y's insert lands with the pre-reauth identification.
        let ops = vec![
            node(y),
            MutationOp::AddEdges {
                base: 2,
                neighbors: vec![1],
                layer: 0,
                edge_type: EdgeType::All,
            },
        ];
        let minted = graph
            .apply_new(UnstampedMutation {
                as_of,
                ops: ops.clone(),
            })
            .unwrap();

        // The record carries the intent verbatim; the void reference is
        // dropped at apply: no edge in either direction.
        assert_eq!(minted.ops, ops);
        assert_eq!(minted.as_of, as_of);
        assert_eq!(graph.get_active_links(&2, 0), Vec::<VectorId>::new());
        assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());

        // Replay re-resolves to the identical graph.
        wal.push(minted);
        let mut replayed = GraphMem::new();
        replayed.insert_apply_all(&wal).unwrap();
        assert_eq!(graph.checksum(), replayed.checksum());
        assert_eq!(replayed.get_active_links(&2, 0), Vec::<VectorId>::new());
    }

    /// Resolution: a `RemoveEdges` record ranked before its target's reauth
    /// must not evict the fresh edge the reauth re-added — the removal names
    /// the old edge, which is already gone. Replay skips it identically.
    #[test]
    fn resolution_skips_removal_of_edge_readded_after_identification() {
        let mut graph = GraphMem::new();
        let mut wal: Vec<GraphMutation> = Vec::new();
        let x0 = VectorId::new(1, 0);
        let x1 = VectorId::new(1, 1);
        let y = VectorId::from_serial_id(2);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: Y -> X.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![
                        node(x0),
                        node(y),
                        MutationOp::AddEdges {
                            base: 2,
                            neighbors: vec![1],
                            layer: 0,
                            edge_type: EdgeType::Base,
                        },
                    ],
                })
                .unwrap(),
        );
        // Compaction ranked Y's neighborhood and chose to evict X here.
        let as_of = graph.last_update_seq_no;
        // seq 2: X reauths and re-links to Y — Y -> X now points at vetted
        // fresh content.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![
                        MutationOp::RemoveNode { id: x1 },
                        node(x1),
                        MutationOp::AddEdges {
                            base: 1,
                            neighbors: vec![2],
                            layer: 0,
                            edge_type: EdgeType::All,
                        },
                    ],
                })
                .unwrap(),
        );
        assert_eq!(graph.get_active_links(&2, 0), vec![x1]);

        // seq 3: the stale-ranked eviction lands; the fresh edge survives.
        let ops = vec![MutationOp::RemoveEdges {
            base: 2,
            neighbors: vec![1],
            layer: 0,
            edge_type: EdgeType::Base,
        }];
        let minted = graph
            .apply_new(UnstampedMutation {
                as_of,
                ops: ops.clone(),
            })
            .unwrap();
        assert_eq!(minted.ops, ops);
        assert_eq!(graph.get_active_links(&2, 0), vec![x1]);

        // Replay re-resolves to the identical graph — the fresh edge
        // survives there too.
        wal.push(minted);
        let mut replayed = GraphMem::new();
        replayed.insert_apply_all(&wal).unwrap();
        assert_eq!(graph.checksum(), replayed.checksum());
        assert_eq!(replayed.get_active_links(&2, 0), vec![x1]);
    }

    /// Resolution: a `base` names the current node with that serial, so an
    /// edge op whose base reauthed after identification applies to the
    /// reauthed node.
    #[test]
    fn resolution_applies_op_whose_base_reauthed_after_identification() {
        let mut graph = GraphMem::new();
        let mut wal: Vec<GraphMutation> = Vec::new();
        let x0 = VectorId::new(1, 0);
        let x1 = VectorId::new(1, 1);
        let z = VectorId::from_serial_id(3);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: X and Z.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![node(x0), node(z)],
                })
                .unwrap(),
        );
        // An edge touch X <-> Z was identified here.
        let as_of = graph.last_update_seq_no;
        // seq 2: X reauths.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![MutationOp::RemoveNode { id: x1 }, node(x1)],
                })
                .unwrap(),
        );
        // seq 3: the touch lands on the reauthed base.
        let ops = vec![MutationOp::AddEdges {
            base: 1,
            neighbors: vec![3],
            layer: 0,
            edge_type: EdgeType::All,
        }];
        let minted = graph
            .apply_new(UnstampedMutation {
                as_of,
                ops: ops.clone(),
            })
            .unwrap();

        assert_eq!(minted.ops, ops);
        assert_eq!(graph.get_active_links(&1, 0), vec![z]);
        assert_eq!(graph.get_active_links(&3, 0), vec![x1]);

        // Replay re-resolves to the identical graph.
        wal.push(minted);
        let mut replayed = GraphMem::new();
        replayed.insert_apply_all(&wal).unwrap();
        assert_eq!(graph.checksum(), replayed.checksum());
        assert_eq!(replayed.get_active_links(&1, 0), vec![z]);
    }

    /// Resolution: an edge op whose base has no current node is dropped
    /// whole — edges must not wire to an absent endpoint.
    #[test]
    fn resolution_drops_op_whose_base_is_absent() {
        let mut graph = GraphMem::new();
        let mut wal: Vec<GraphMutation> = Vec::new();
        let x0 = VectorId::new(1, 0);
        let z = VectorId::from_serial_id(3);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: X and Z.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![node(x0), node(z)],
                })
                .unwrap(),
        );
        // An edge touch X <-> Z was identified here.
        let as_of = graph.last_update_seq_no;
        // seq 2: X is removed.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![MutationOp::RemoveNode { id: x0 }],
                })
                .unwrap(),
        );
        // seq 3: the touch lands with its base gone and is void.
        let ops = vec![MutationOp::AddEdges {
            base: 1,
            neighbors: vec![3],
            layer: 0,
            edge_type: EdgeType::All,
        }];
        let minted = graph
            .apply_new(UnstampedMutation {
                as_of,
                ops: ops.clone(),
            })
            .unwrap();

        assert_eq!(minted.ops, ops);
        assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
        assert_eq!(graph.get_active_links(&3, 0), Vec::<VectorId>::new());

        // Replay re-resolves to the identical graph.
        wal.push(minted);
        let mut replayed = GraphMem::new();
        replayed.insert_apply_all(&wal).unwrap();
        assert_eq!(graph.checksum(), replayed.checksum());
        assert_eq!(replayed.get_active_links(&3, 0), Vec::<VectorId>::new());
    }

    /// Resolution: a reauth plan's search almost always finds the node's own
    /// old version, and the serial collapse turns it into a self-reference.
    /// It was identified against the content the reauth replaces, so it must
    /// not survive as a self-edge — while legitimate neighbors do.
    #[test]
    fn resolution_drops_reauth_self_reference() {
        let mut graph = GraphMem::new();
        let mut wal: Vec<GraphMutation> = Vec::new();
        let s0 = VectorId::new(1, 0);
        let s1 = VectorId::new(1, 1);
        let a = VectorId::from_serial_id(2);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        // seq 1: S and anchor A, linked.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![
                        node(s0),
                        node(a),
                        MutationOp::AddEdges {
                            base: 1,
                            neighbors: vec![2],
                            layer: 0,
                            edge_type: EdgeType::All,
                        },
                    ],
                })
                .unwrap(),
        );
        // The reauth's search identified [S(old), A] here.
        let as_of = graph.last_update_seq_no;
        // seq 2: teardown, its own record (the production shape).
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of: graph.last_update_seq_no,
                    ops: vec![MutationOp::RemoveNode { id: s1 }],
                })
                .unwrap(),
        );
        // seq 3: re-insert wiring the searched links, self-echo included.
        wal.push(
            graph
                .apply_new(UnstampedMutation {
                    as_of,
                    ops: vec![
                        node(s1),
                        MutationOp::AddEdges {
                            base: 1,
                            neighbors: vec![1, 2],
                            layer: 0,
                            edge_type: EdgeType::All,
                        },
                    ],
                })
                .unwrap(),
        );

        // No self-edge, physically or actively; the real neighbor survives
        // in both directions.
        assert_eq!(graph.get_raw_links(&1, 0), &[2u32]);
        assert_eq!(graph.get_active_links(&1, 0), vec![a]);
        assert_eq!(graph.get_active_links(&2, 0), vec![s1]);

        // Replay re-resolves to the identical graph.
        let mut replayed = GraphMem::new();
        replayed.insert_apply_all(&wal).unwrap();
        assert_eq!(graph.checksum(), replayed.checksum());
        assert_eq!(replayed.get_raw_links(&1, 0), &[2u32]);
    }

    /// Read-path skip: `get_active_links` omits a content-stale neighbor even
    /// when the neighborhood was never touched after the reauth, so the
    /// physical edge is still present.
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
                as_of: 0,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // Asymmetric a -> b; a's neighborhood certified at seq 2.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(graph.get_active_links(&1, 0), vec![b]);

        // Reauth b (content[b] -> 3) without touching a.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                as_of: 2,
                ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
            })
            .unwrap();

        // Physical edge survives (a untouched), but read-path skips it.
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );
        assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
    }

    /// Read-path liveness: `get_active_links` drops a dangling edge to a
    /// removed node. The physical edge survives `RemoveNode` (no backlink
    /// cleanup); the removed node has no content-clock entry, so `is_active`
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
                as_of: 0,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        // Asymmetric a -> b so b's removal can't clean a's back-edge.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        // Remove b (no re-insert): content[b] is dropped.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                as_of: 2,
                ops: vec![MutationOp::RemoveNode { id: b }],
            })
            .unwrap();

        // Physical edge a -> b survives, but read-path skips the dead node.
        assert_eq!(
            graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
            vec![2u32]
        );
        assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
    }

    /// Read-path selectivity: a live neighbor survives while a content-stale
    /// neighbor is dropped. Guards against `get_active_links` blanket-emptying
    /// (e.g. an inverted `is_active`), which the single-neighbor tests above
    /// would not catch. `c` carries a non-zero version, pinning that the
    /// surviving link resolves to the version its `AddNode` carried.
    #[test]
    fn get_active_links_keeps_live_drops_stale() {
        let mut graph = GraphMem::new();
        let a = VectorId::from_serial_id(1);
        let b = VectorId::from_serial_id(2);
        let c = VectorId::new(3, 5);
        let node = |id| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };

        graph
            .insert_apply(&GraphMutation {
                seq_no: 1,
                as_of: 0,
                ops: vec![node(a), node(b), node(c)],
            })
            .unwrap();
        // a -> [b, c], certified at seq 2.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2, 3],
                    edge_type: EdgeType::Base,
                }],
            })
            .unwrap();
        assert_eq!(graph.get_active_links(&1, 0), vec![b, c]);

        // Reauth only b (content[b] -> 3); a and c untouched.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                as_of: 2,
                ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
            })
            .unwrap();

        // c (content[c]=1 <= 2) survives at its inserted version; b
        // (content[b]=3 > 2) is skipped.
        assert_eq!(graph.get_active_links(&1, 0), vec![c]);
        assert_eq!(graph.vector_id_of(3), Some(c));
        assert_eq!(graph.vector_id_of(4), None, "never-inserted serial");
    }

    /// The incremental `node_init_hash` (mutation apply) must match the bulk
    /// recompute (`from_parts`, via deserialize) after a mutation history that
    /// includes a reauth.
    #[test]
    fn incremental_checksum_matches_bulk_after_reauth() {
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
                as_of: 0,
                ops: vec![node(a), node(b)],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
                    edge_type: EdgeType::All,
                }],
            })
            .unwrap();
        // Reauth b: RemoveNode + AddNode, incrementally updating node_init_hash.
        graph
            .insert_apply(&GraphMutation {
                seq_no: 3,
                as_of: 2,
                ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
            })
            .unwrap();

        // bincode round trip: Deserialize routes through from_parts, which
        // recomputes node_init_hash in bulk.
        let buf = bincode::serialize(&graph).unwrap();
        let bulk: GraphMem = bincode::deserialize(&buf).unwrap();

        assert_eq!(
            graph.checksum(),
            bulk.checksum(),
            "incremental node_init_hash drifted from bulk recompute"
        );
    }

    /// A graph aged through insert, mixed edge types, reauth, and delete
    /// round-trips through the current pair format, and its incremental
    /// checksum matches a bulk recompute.
    #[tokio::test]
    async fn graph_v5_evolving_roundtrip() {
        use crate::utils::serialization::graph::{
            read_graph_pair, write_graph_pair_current, GraphFormat,
        };
        let node = |id: VectorId| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };
        let v = VectorId::from_serial_id;
        let mut g = GraphMem::new();

        // Build: node 1 spans layers 0,1 (entry point at L1); 2..=4 in layer 0.
        g.insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![
                MutationOp::AddNode {
                    id: v(1),
                    height: 2,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                node(v(2)),
                node(v(3)),
                node(v(4)),
            ],
        })
        .unwrap();
        assert_eq!(g.get_num_layers(), 2);
        assert_eq!(g.get_entry_points(), Some(vec![1]));

        // Mixed edge types at layer 0: All(1<->{2,3}) + Base(3->4).
        g.insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![
                MutationOp::AddEdges {
                    base: 1,
                    neighbors: vec![2, 3],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                MutationOp::AddEdges {
                    base: 3,
                    neighbors: vec![4],
                    layer: 0,
                    edge_type: EdgeType::Base,
                },
            ],
        })
        .unwrap();
        assert_eq!(g.get_active_links(&1, 0), vec![v(2), v(3)]);
        assert_eq!(g.get_active_links(&2, 0), vec![v(1)]);

        // Reauth node 3 (bumps its content clock) and delete node 4, so the aged
        // graph carries a non-uniform content clock and a removed node.
        g.insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: v(3) }, node(v(3))],
        })
        .unwrap();
        g.insert_apply(&GraphMutation {
            seq_no: 4,
            as_of: 3,
            ops: vec![MutationOp::RemoveNode { id: v(4) }],
        })
        .unwrap();
        assert_eq!(g.last_update_seq_no, 4);

        // Incremental checksum matches a bulk (from_parts) recompute over the
        // aged graph, and it round-trips through the production pair format.
        let bulk: GraphMem = bincode::deserialize(&bincode::serialize(&g).unwrap()).unwrap();
        assert_eq!(
            g.checksum(),
            bulk.checksum(),
            "incremental checksum drifted"
        );

        let mut pair_buf = Vec::new();
        write_graph_pair_current(&mut pair_buf, [g.clone(), g.clone()]).unwrap();
        let pair =
            read_graph_pair(&mut std::io::Cursor::new(&pair_buf), GraphFormat::Current).unwrap();
        for restored in &pair {
            assert_eq!(*restored, g, "pair round-trip changed the aged graph");
        }
    }

    /// `RemoveEdges` fused retain: one teardown drops the explicitly-named
    /// edge AND sweeps a content-stale sibling in a single re-stamp, on both
    /// the base-list half (EdgeType::Base) and the target-loop back half
    /// (EdgeType::All). The sweep is not reflected in the returned op list.
    #[tokio::test]
    async fn remove_edges_op_fused_retain_drops_explicit_and_sweeps_stale() {
        let node = |id: VectorId| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };
        let v = VectorId::from_serial_id;
        let mut g = GraphMem::new();
        g.apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: (1..=6).map(|i| node(v(i))).collect(),
        })
        .unwrap();

        // ── EdgeType::Base, forward-half retain. 1 -> {2,3} forward-only, then
        // reauth 3 so 1->3 is content-stale while 1's list is untouched.
        g.apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                neighbors: vec![2, 3],
                layer: 0,
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
        g.apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![MutationOp::RemoveNode { id: v(3) }, node(v(3))],
        })
        .unwrap();
        assert_eq!(g.get_raw_links(&1, 0), &[2u32, 3u32], "stale 3 still raw");
        assert_eq!(g.get_active_links(&1, 0), vec![v(2)], "3 masked as stale");

        // RemoveEdges Base(1 -/-> 2): retain drops explicit 2 AND stale 3; the
        // returned op list carries the intent unchanged.
        let teardown = MutationOp::RemoveEdges {
            base: 1,
            neighbors: vec![2],
            layer: 0,
            edge_type: EdgeType::Base,
        };
        let minted = g
            .apply_new(UnstampedMutation {
                as_of: g.last_update_seq_no,
                ops: vec![teardown.clone()],
            })
            .unwrap();
        assert_eq!(
            g.get_raw_links(&1, 0),
            &[] as &[u32],
            "explicit 2 removed AND content-stale 3 swept in one retain"
        );
        assert_eq!(minted.ops, vec![teardown], "ops returned unchanged");

        // ── EdgeType::All, back-half (target-loop) retain. 4<->5 symmetric plus
        // asymmetric 5->6; reauth 6 so 5->6 is stale.
        g.apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![
                MutationOp::AddEdges {
                    base: 4,
                    neighbors: vec![5],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                MutationOp::AddEdges {
                    base: 5,
                    neighbors: vec![6],
                    layer: 0,
                    edge_type: EdgeType::Base,
                },
            ],
        })
        .unwrap();
        g.apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![MutationOp::RemoveNode { id: v(6) }, node(v(6))],
        })
        .unwrap();
        assert_eq!(g.get_raw_links(&5, 0), &[4u32, 6u32], "stale 6 still raw");

        // RemoveEdges All(4 -/- 5): forward half empties 4; back-half retain on
        // target 5 drops explicit 4 AND stale 6.
        let teardown = MutationOp::RemoveEdges {
            base: 4,
            neighbors: vec![5],
            layer: 0,
            edge_type: EdgeType::All,
        };
        let minted = g
            .apply_new(UnstampedMutation {
                as_of: g.last_update_seq_no,
                ops: vec![teardown.clone()],
            })
            .unwrap();
        assert_eq!(
            g.get_raw_links(&4, 0),
            &[] as &[u32],
            "forward half drops 5"
        );
        assert_eq!(
            g.get_raw_links(&5, 0),
            &[] as &[u32],
            "back-half retain drops explicit 4 AND content-stale 6"
        );
        assert_eq!(minted.ops, vec![teardown], "ops returned unchanged");
    }

    /// A stream minted by `apply_new` replays via `insert_apply_all` onto a
    /// fresh graph to the identical state — checksum included. Exercises:
    /// reauth whose re-wiring touches a neighborhood holding its own stale
    /// back-edge, deletion followed by a touch sweeping the dangling edge,
    /// both `RemoveEdges` fused-retain halves (Base forward, All back), a
    /// compaction-shaped `RemoveEdges`, multi-layer edges, and entry-point
    /// churn.
    #[test]
    fn minted_stream_replays_to_identical_graph() {
        fn mint(g: &mut GraphMem, wal: &mut Vec<GraphMutation>, ops: Vec<MutationOp>) {
            wal.push(
                g.apply_new(UnstampedMutation {
                    as_of: g.last_update_seq_no,
                    ops,
                })
                .unwrap(),
            );
        }
        let node = |s: u32| MutationOp::AddNode {
            id: VectorId::from_serial_id(s),
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };
        // Node 1 spans layers 0,1 and is the entry point; re-minted on reauth.
        let node1 = || MutationOp::AddNode {
            id: VectorId::from_serial_id(1),
            height: 2,
            update_ep: UpdateEntryPoint::Append { layer: 1 },
        };
        let all = |base: u32, neighbors: Vec<u32>, layer: usize| MutationOp::AddEdges {
            base,
            neighbors,
            layer,
            edge_type: EdgeType::All,
        };

        let mut l = GraphMem::new();
        let mut wal = Vec::new();

        // seq 1: nodes 1..=7 (1 and 5 span layer 1).
        let mut ops = vec![node1()];
        ops.extend((2..=7).map(node));
        ops[4] = MutationOp::AddNode {
            id: VectorId::from_serial_id(5),
            height: 2,
            update_ep: UpdateEntryPoint::False,
        };
        mint(&mut l, &mut wal, ops);
        // seq 2: 1<->2 at layer 0, 1<->5 at layer 1.
        mint(
            &mut l,
            &mut wal,
            vec![all(1, vec![2], 0), all(1, vec![5], 1)],
        );
        // seq 3: reauth 1, phase one — node teardown (2 and 5 keep stale
        // back-edges to 1; the entry point is dropped).
        mint(
            &mut l,
            &mut wal,
            vec![MutationOp::RemoveNode {
                id: VectorId::from_serial_id(1),
            }],
        );
        // seq 4: reauth 1, phase two — KEY fixture: the re-wiring back-halves
        // touch 2 (layer 0) and 5 (layer 1), each holding the stale edge to 1
        // that the same op re-adds (drop-then-re-add within one apply).
        mint(
            &mut l,
            &mut wal,
            vec![node1(), all(1, vec![2], 0), all(1, vec![5], 1)],
        );
        assert_eq!(
            wal[3].ops,
            vec![node1(), all(1, vec![2], 0), all(1, vec![5], 1)],
            "minted ops are the intent, unmodified"
        );

        // seq 5: pure deletion of 2 — node 1 keeps a dangling 1->2 edge.
        mint(
            &mut l,
            &mut wal,
            vec![MutationOp::RemoveNode {
                id: VectorId::from_serial_id(2),
            }],
        );
        // seq 6: 3<->1 — the back-half touch of 1 sweeps the dangling 2.
        mint(&mut l, &mut wal, vec![all(3, vec![1], 0)]);
        // seq 7: forward-only fan-out 5 -> {6,7} at layer 0.
        mint(
            &mut l,
            &mut wal,
            vec![MutationOp::AddEdges {
                base: 5,
                neighbors: vec![6, 7],
                layer: 0,
                edge_type: EdgeType::Base,
            }],
        );
        // seq 8: reauth 6 in one mutation (no edges reference 6 here).
        mint(
            &mut l,
            &mut wal,
            vec![
                MutationOp::RemoveNode {
                    id: VectorId::from_serial_id(6),
                },
                node(6),
            ],
        );
        // seq 9: compaction-shaped RemoveEdges (Base): drops named 7 and sweeps
        // stale 6 from 5's list in the fused forward retain.
        mint(
            &mut l,
            &mut wal,
            vec![MutationOp::RemoveEdges {
                base: 5,
                neighbors: vec![7],
                layer: 0,
                edge_type: EdgeType::Base,
            }],
        );
        // seq 10: 4<->3.
        mint(&mut l, &mut wal, vec![all(4, vec![3], 0)]);
        // seq 11: reauth 1 again — 3's list (certified at seq 10) holds a now
        // stale edge to 1.
        mint(
            &mut l,
            &mut wal,
            vec![
                MutationOp::RemoveNode {
                    id: VectorId::from_serial_id(1),
                },
                node1(),
            ],
        );
        // seq 12: RemoveEdges All 4 -/- 3: the back-half target retain on 3
        // drops named 4 and sweeps stale 1.
        mint(
            &mut l,
            &mut wal,
            vec![MutationOp::RemoveEdges {
                base: 4,
                neighbors: vec![3],
                layer: 0,
                edge_type: EdgeType::All,
            }],
        );

        // Only the two explicit RemoveEdges (seq 9, 12) appear in the stream;
        // filter drops are never recorded.
        let remove_edges = wal
            .iter()
            .flat_map(|m| m.ops.iter())
            .filter(|op| matches!(op, MutationOp::RemoveEdges { .. }))
            .count();
        assert_eq!(remove_edges, 2, "ops must record intent only");

        let mut r = GraphMem::new();
        r.insert_apply_all(&wal).unwrap();
        assert_eq!(l.checksum(), r.checksum(), "replay diverged from mint");
        assert_eq!(l, r, "replay diverged from mint");
    }

    /// Edge ops never perturb the content clock: a node's `NodeInit` moves
    /// only on its own `AddNode`/`RemoveNode`.
    #[tokio::test]
    async fn edge_ops_do_not_perturb_content_clock() {
        let node = |id: VectorId| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };
        let v = VectorId::from_serial_id;
        let mut g = GraphMem::new();
        // Node 2 carries a non-zero version so the clock has real content.
        g.insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![node(v(1)), node(VectorId::new(2, 7)), node(v(3))],
        })
        .unwrap();
        let clock_before = g.node_init.clone();

        g.insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                neighbors: vec![2, 3],
                layer: 0,
                edge_type: EdgeType::All,
            }],
        })
        .unwrap();
        assert_eq!(g.get_raw_links(&1, 0), &[2u32, 3u32]);
        assert_eq!(
            g.get_raw_links(&2, 0),
            &[1u32],
            "back-edge keyed on serial 2"
        );
        assert_eq!(
            g.node_init, clock_before,
            "AddEdges must not perturb the content clock"
        );

        g.insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveEdges {
                base: 1,
                neighbors: vec![2],
                layer: 0,
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
        assert_eq!(g.get_raw_links(&1, 0), &[3u32]);
        assert_eq!(
            g.node_init, clock_before,
            "RemoveEdges must not perturb the content clock"
        );
    }

    /// Deleting the sole entry point: RemoveNode drops it from every layer,
    /// the entry-point list, and the content clock. Backlinks to it linger in
    /// raw but are masked by is_active; get_temporary_entry_point falls back
    /// to the min-serial node of the top non-empty layer.
    #[tokio::test]
    async fn remove_node_of_entry_point_falls_back_to_min_serial() {
        let node = |id: VectorId| MutationOp::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        };
        let v = VectorId::from_serial_id;
        let mut g = GraphMem::new();
        // Node 1 spans layers 0,1 and is the sole entry point (at L1); 2,3 in L0.
        g.insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![
                MutationOp::AddNode {
                    id: v(1),
                    height: 2,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                node(v(2)),
                node(v(3)),
            ],
        })
        .unwrap();
        // Symmetric backlinks 1<->2, 1<->3 at layer 0.
        g.insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                neighbors: vec![2, 3],
                layer: 0,
                edge_type: EdgeType::All,
            }],
        })
        .unwrap();
        assert_eq!(g.get_entry_points(), Some(vec![1]));

        g.insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: v(1) }],
        })
        .unwrap();

        assert!(g.layers[0].get_links(&1).is_none(), "gone at layer 0");
        assert!(g.layers[1].get_links(&1).is_none(), "gone at layer 1");
        // Backlinks 2->1 / 3->1 linger physically but are masked at read: 1 is
        // gone from the content clock, so is_active hides the dangling edge.
        assert_eq!(g.get_raw_links(&2, 0), &[1u32], "2->1 lingers in raw");
        assert!(
            g.get_active_links(&2, 0).is_empty(),
            "2->1 masked by is_active"
        );
        assert_eq!(g.get_raw_links(&3, 0), &[1u32], "3->1 lingers in raw");
        assert!(
            g.get_active_links(&3, 0).is_empty(),
            "3->1 masked by is_active"
        );
        assert_eq!(g.get_entry_points(), None, "entry-point list emptied");
        // L1 now empty, so fall back to min serial of L0.
        assert_eq!(g.get_temporary_entry_point(), Some((2u32, 0)));
        assert!(!g.node_init.contains_key(&1), "content[1] dropped");
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
                as_of: 0,
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
                as_of: 1,
                ops: vec![MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2, 3],
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
                as_of: 0,
                ops: vec![
                    MutationOp::AddNode {
                        id: a,
                        height: 1,
                        update_ep: UpdateEntryPoint::Append { layer: 1 },
                    },
                    MutationOp::AddEdges {
                        base: 1,
                        layer: 0,
                        neighbors: vec![2, 3],
                        edge_type: EdgeType::Base,
                    },
                    MutationOp::AddNode {
                        id: b,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddEdges {
                        base: 2,
                        layer: 0,
                        neighbors: vec![1],
                        edge_type: EdgeType::Base,
                    },
                    MutationOp::AddNode {
                        id: c,
                        height: 1,
                        update_ep: UpdateEntryPoint::False,
                    },
                    MutationOp::AddEdges {
                        base: 3,
                        layer: 0,
                        neighbors: vec![1],
                        edge_type: EdgeType::Base,
                    },
                ],
            })
            .unwrap();
        graph
            .insert_apply(&GraphMutation {
                seq_no: 2,
                as_of: 1,
                ops: vec![MutationOp::RemoveEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
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
                as_of: 0,
                ops: vec![
                    // Listed first: an edge op that references a node not yet created.
                    MutationOp::AddEdges {
                        base: 1,
                        layer: 0,
                        neighbors: vec![2],
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
            as_of: 0,
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
            as_of: 4,
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
            as_of: 8,
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
                as_of: 0,
                ops: vec![MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 0 },
                }],
            },
            // Equal seq_no — should fail.
            GraphMutation {
                seq_no: 1,
                as_of: 0,
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
            // Select the top-k by distance, then compare as a set: neighbor lists
            // are stored serial-sorted (canonical for the set-hash), so distance
            // order is not preserved — only membership of the correct k matters.
            let mut expected: Vec<SerialId> = dists.into_iter().take(k).map(|(j, _)| j).collect();
            expected.sort_unstable();
            assert_eq!(&nbhd.neighbors, &expected, "key {key}");
        }
    }
}
