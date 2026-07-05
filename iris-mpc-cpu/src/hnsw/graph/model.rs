//! Executable specification of the graph the WAL contracts with.
//!
//! [`ModelGraph`] is the abstract graph: versioned nodes, entry points, plain
//! per-layer edge sets — no timestamps, no staleness. Ops are set arithmetic;
//! removal is eager: removing or re-inserting a node deletes every incident
//! edge, both directions, immediately. `GraphMem` implements the same
//! semantics lazily; the refinement test drives both with generated mutation
//! streams and asserts their observations agree after every mutation.
//!
//! Spec preconditions (asserted here, upheld by production mints): `AddEdges`
//! endpoints are live at apply time (causal construction); `AddNode` targets
//! a non-live serial (reauth = `RemoveNode` + `AddNode` in one mutation).

use super::mutation::{EdgeType, MutationOp, UpdateEntryPoint};
use iris_mpc_common::{SerialId, VersionId};
use std::collections::{BTreeMap, BTreeSet};

/// Reference implementation the production graph is tested against; keep it
/// free of laziness, clocks, and performance concerns.
#[derive(Default)]
pub struct ModelGraph {
    /// Live nodes and the version their insertion carried.
    nodes: BTreeMap<SerialId, VersionId>,
    /// Per-layer adjacency: node -> outgoing edge set.
    layers: Vec<BTreeMap<SerialId, BTreeSet<SerialId>>>,
    /// Entry points as (node, layer), in insertion order.
    entry_points: Vec<(SerialId, usize)>,
}

impl ModelGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply one mutation: node ops first, then edge ops, mirroring the
    /// two-pass order of `GraphMem::apply_ops`.
    pub fn apply(&mut self, ops: &[MutationOp]) {
        for op in ops {
            match op {
                MutationOp::RemoveNode { id } => self.remove_node(id.serial_id()),
                MutationOp::AddNode {
                    id,
                    height,
                    update_ep,
                } => {
                    let sid = id.serial_id();
                    assert!(
                        !self.nodes.contains_key(&sid),
                        "spec precondition: AddNode on live serial {sid} (reauth = RemoveNode + AddNode)"
                    );
                    if let UpdateEntryPoint::Append { layer } = update_ep {
                        self.entry_points.push((sid, *layer));
                    }
                    if self.layers.len() < *height {
                        self.layers.resize(*height, BTreeMap::new());
                    }
                    for layer in self.layers.iter_mut().take(*height) {
                        layer.insert(sid, BTreeSet::new());
                    }
                    self.nodes.insert(sid, id.version_id());
                }
                MutationOp::AddEdges { .. } | MutationOp::RemoveEdges { .. } => {}
            }
        }

        for op in ops {
            match op {
                MutationOp::AddNode { .. } | MutationOp::RemoveNode { .. } => {}
                MutationOp::AddEdges {
                    base,
                    neighbors,
                    layer,
                    edge_type,
                } => {
                    assert!(
                        neighbors.iter().all(|z| self.nodes.contains_key(z)),
                        "spec precondition: AddEdges target not live (causal construction)"
                    );
                    if self.layers.len() < layer + 1 {
                        self.layers.resize(layer + 1, BTreeMap::new());
                    }
                    let lmap = &mut self.layers[*layer];
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        // Base absent from the layer: skip the outgoing half.
                        if let Some(set) = lmap.get_mut(base) {
                            set.extend(neighbors.iter().copied());
                        }
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for target in neighbors {
                            // Target absent from the layer: skip the back-edge.
                            if let Some(set) = lmap.get_mut(target) {
                                set.insert(*base);
                            }
                        }
                    }
                }
                MutationOp::RemoveEdges {
                    base,
                    neighbors,
                    layer,
                    edge_type,
                } => {
                    if *layer >= self.layers.len() {
                        continue;
                    }
                    let lmap = &mut self.layers[*layer];
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        if let Some(set) = lmap.get_mut(base) {
                            for target in neighbors {
                                set.remove(target);
                            }
                        }
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for target in neighbors {
                            if let Some(set) = lmap.get_mut(target) {
                                set.remove(base);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Eager removal: the node, its entry points, and every edge incident to
    /// it — both directions, all layers — disappear in one transition.
    fn remove_node(&mut self, sid: SerialId) {
        self.nodes.remove(&sid);
        self.entry_points.retain(|(point, _)| *point != sid);
        for layer in &mut self.layers {
            layer.remove(&sid);
            for set in layer.values_mut() {
                set.remove(&sid);
            }
        }
    }

    /* ---------------------------- observations --------------------------- */

    pub fn version_of(&self, sid: SerialId) -> Option<VersionId> {
        self.nodes.get(&sid).copied()
    }

    pub fn is_member(&self, sid: SerialId, layer: usize) -> bool {
        self.layers.get(layer).is_some_and(|l| l.contains_key(&sid))
    }

    /// Outgoing edges of `sid` at `layer` as (serial, version), sorted by
    /// serial. Empty if `sid` is not a member of the layer.
    pub fn active_links(&self, sid: SerialId, layer: usize) -> Vec<(SerialId, VersionId)> {
        let Some(set) = self.layers.get(layer).and_then(|l| l.get(&sid)) else {
            return Vec::new();
        };
        set.iter().map(|z| (*z, self.nodes[z])).collect()
    }

    pub fn entry_points(&self) -> &[(SerialId, usize)] {
        &self.entry_points
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::graph::layered_graph::GraphMem;
    use crate::hnsw::graph::mutation::{GraphMutation, UnstampedMutation};
    use iris_mpc_common::VectorId;
    use rand::{seq::SliceRandom, Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    /// Serial universe the generator draws from; small enough that removals,
    /// resurrections, and re-wirings collide often.
    const UNIVERSE: std::ops::RangeInclusive<u32> = 1..=20;
    const MAX_HEIGHT: usize = 3;

    /// Every observation the spec defines must agree between the production
    /// graph and the model.
    fn assert_refines(g: &GraphMem, m: &ModelGraph, step: usize) {
        for s in UNIVERSE {
            assert_eq!(
                g.vector_id_of(s),
                m.version_of(s).map(|v| VectorId::new(s, v)),
                "step {step}: version of serial {s}"
            );
        }
        let g_eps: Vec<(SerialId, usize)> =
            g.entry_points.iter().map(|e| (e.point, e.layer)).collect();
        assert_eq!(g_eps, m.entry_points(), "step {step}: entry points");
        let layers = m.num_layers().max(g.layers.len());
        for lc in 0..layers {
            for s in UNIVERSE {
                let member_g = g.layers.get(lc).is_some_and(|l| l.get_links(&s).is_some());
                assert_eq!(
                    member_g,
                    m.is_member(s, lc),
                    "step {step}: membership of {s} at layer {lc}"
                );
                let got: Vec<(SerialId, VersionId)> = g
                    .get_active_links(&s, lc)
                    .iter()
                    .map(|v| (v.serial_id(), v.version_id()))
                    .collect();
                assert_eq!(
                    got,
                    m.active_links(s, lc),
                    "step {step}: active links of {s} at layer {lc}"
                );
            }
        }
    }

    /// Random serials that are live in the model and members of `layer`,
    /// excluding `not`; at most `k`, without replacement.
    fn pick_members(
        m: &ModelGraph,
        layer: usize,
        not: SerialId,
        k: usize,
        rng: &mut ChaCha8Rng,
    ) -> Vec<SerialId> {
        let mut members: Vec<SerialId> = UNIVERSE
            .filter(|s| *s != not && m.is_member(*s, layer))
            .collect();
        members.shuffle(rng);
        members.truncate(k);
        members
    }

    /// Ops inserting `s` at `height` and wiring it (EdgeType::All) to a few
    /// members of each layer it joins — the shape of a production insert.
    fn insert_ops(
        m: &ModelGraph,
        s: SerialId,
        version: VersionId,
        height: usize,
        rng: &mut ChaCha8Rng,
    ) -> Vec<MutationOp> {
        let update_ep = if height > m.num_layers() || rng.gen_bool(0.05) {
            UpdateEntryPoint::Append { layer: height - 1 }
        } else {
            UpdateEntryPoint::False
        };
        let mut ops = vec![MutationOp::AddNode {
            id: VectorId::new(s, version),
            height,
            update_ep,
        }];
        for layer in 0..height {
            let targets = pick_members(m, layer, s, rng.gen_range(1..=4), rng);
            if !targets.is_empty() {
                ops.push(MutationOp::AddEdges {
                    base: s,
                    neighbors: targets,
                    layer,
                    edge_type: EdgeType::All,
                });
            }
        }
        ops
    }

    /// Drive `GraphMem` and the model with the same generated mutation
    /// stream — inserts, reauths, deletions, edge touches, compaction-shaped
    /// removals — checking observation agreement after every mutation; then
    /// the recorded stream must replay onto a fresh graph to identical state.
    #[test]
    fn graphmem_refines_the_model() {
        for seed in 0..8u64 {
            let rng = &mut ChaCha8Rng::seed_from_u64(seed);
            let mut g = GraphMem::new();
            let mut m = ModelGraph::new();
            let mut wal: Vec<GraphMutation> = Vec::new();

            for step in 0..300 {
                let live: Vec<SerialId> = UNIVERSE.filter(|s| m.version_of(*s).is_some()).collect();
                let ops: Vec<MutationOp> = match rng.gen_range(0..100) {
                    // Insert a non-live serial (fresh or resurrected).
                    0..=29 => {
                        let free: Vec<SerialId> =
                            UNIVERSE.filter(|s| m.version_of(*s).is_none()).collect();
                        match free.choose(rng) {
                            Some(&s) => insert_ops(
                                &m,
                                s,
                                rng.gen_range(0..4),
                                rng.gen_range(1..=MAX_HEIGHT),
                                rng,
                            ),
                            None => continue,
                        }
                    }
                    // Reauth: teardown + re-insert + re-wire in one mutation.
                    30..=49 => match live.choose(rng) {
                        Some(&s) => {
                            let version = m.version_of(s).unwrap() + 1;
                            let mut ops = vec![MutationOp::RemoveNode {
                                id: VectorId::new(s, version),
                            }];
                            ops.extend(insert_ops(
                                &m,
                                s,
                                version,
                                rng.gen_range(1..=MAX_HEIGHT),
                                rng,
                            ));
                            ops
                        }
                        None => continue,
                    },
                    // Deletion.
                    50..=64 => match live.choose(rng) {
                        Some(&s) => vec![MutationOp::RemoveNode {
                            id: VectorId::new(s, m.version_of(s).unwrap()),
                        }],
                        None => continue,
                    },
                    // Edge touch: wire a live base to live targets.
                    65..=84 => {
                        let layer = rng.gen_range(0..MAX_HEIGHT);
                        let bases = pick_members(&m, layer, 0, 1, rng);
                        let Some(&base) = bases.first() else { continue };
                        let targets = pick_members(&m, layer, base, rng.gen_range(1..=3), rng);
                        if targets.is_empty() {
                            continue;
                        }
                        let edge_type = [EdgeType::Base, EdgeType::Neighbors, EdgeType::All]
                            .choose(rng)
                            .unwrap()
                            .clone();
                        vec![MutationOp::AddEdges {
                            base,
                            neighbors: targets,
                            layer,
                            edge_type,
                        }]
                    }
                    // Compaction-shaped RemoveEdges (named serials need not be
                    // present or live).
                    _ => {
                        let layer = rng.gen_range(0..MAX_HEIGHT);
                        let bases = pick_members(&m, layer, 0, 1, rng);
                        let Some(&base) = bases.first() else { continue };
                        let mut targets: Vec<SerialId> = m
                            .active_links(base, layer)
                            .iter()
                            .map(|(s, _)| *s)
                            .collect();
                        targets.shuffle(rng);
                        targets.truncate(2);
                        targets.push(*UNIVERSE.collect::<Vec<_>>().choose(rng).unwrap());
                        targets.sort_unstable();
                        targets.dedup();
                        targets.retain(|t| *t != base);
                        if targets.is_empty() {
                            continue;
                        }
                        let edge_type =
                            [EdgeType::Base, EdgeType::All].choose(rng).unwrap().clone();
                        vec![MutationOp::RemoveEdges {
                            base,
                            neighbors: targets,
                            layer,
                            edge_type,
                        }]
                    }
                };

                wal.push(g.apply_new(UnstampedMutation { ops: ops.clone() }).unwrap());
                m.apply(&ops);
                assert_refines(&g, &m, step);
            }

            let mut replayed = GraphMem::new();
            replayed.insert_apply_all(&wal).unwrap();
            assert_eq!(
                g.checksum(),
                replayed.checksum(),
                "seed {seed}: replay checksum"
            );
            assert_eq!(g, replayed, "seed {seed}: replay state");
            assert_refines(&replayed, &m, usize::MAX);
        }
    }
}
