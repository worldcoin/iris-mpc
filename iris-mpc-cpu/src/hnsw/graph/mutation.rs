use iris_mpc_common::VectorId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphMutation {
    pub seq_no: u64,
    pub ops: Vec<MutationOp>,
}

/// A graph mutation that has not yet been assigned a sequence number.
///
/// Holds the ops describing *what* to change without committing to *where* in
/// the mutation sequence the change lands. The only way to turn one into a
/// stamped [`GraphMutation`] is [`crate::hnsw::GraphMem::apply_new`], which
/// assigns the sequence number and applies it in a single step — so a sequence
/// number can never be minted in-process without immediately advancing the
/// graph.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct UnstampedMutation {
    pub ops: Vec<MutationOp>,
}

// NOTE: if a new version of any mutation is needed (ex: InsertNodeV2) such that
// the new variant would behave differently than before and it is desired to
// still process old variants apppropriately, simply add the new variant to the
// END of MutationOp. If the new variant is added to the end then bincode can
// deserialize it correctly. Adding a new variant in between existing ones will
// cause bincode to deserialize the old version of MutationOp with garbage data
// for its fields.
//
/// Represents a diff to apply to an existing graph.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationOp {
    AddNode {
        id: VectorId,
        /// Number of real graph layers this node is included in. The node will
        /// be present in layers `0..height`.
        height: usize,
        update_ep: UpdateEntryPoint,
    },
    RemoveNode {
        id: VectorId,
    },
    AddEdges {
        base: VectorId,
        neighbors: Vec<VectorId>,
        layer: usize,
        edge_type: EdgeType,
    },
    RemoveEdges {
        base: VectorId,
        neighbors: Vec<VectorId>,
        layer: usize,
        edge_type: EdgeType,
    },
}

impl std::fmt::Debug for MutationOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RemoveNode { id } => f.debug_struct("RemoveNode").field("id", id).finish(),
            Self::AddNode {
                id,
                height,
                update_ep,
            } => f
                .debug_struct("AddNode")
                .field("id", id)
                .field("height", height)
                .field("update_ep", update_ep)
                .finish(),
            Self::AddEdges {
                base,
                layer,
                edge_type,
                ..
            } => f
                .debug_struct("AddEdges")
                .field("base", base)
                .field("layer", layer)
                .field("edge_type", edge_type)
                .finish(),
            Self::RemoveEdges {
                base,
                layer,
                edge_type,
                ..
            } => f
                .debug_struct("RemoveEdges")
                .field("base", base)
                .field("layer", layer)
                .field("edge_type", edge_type)
                .finish(),
        }
    }
}

impl UnstampedMutation {
    /// Subset of `updated_neighborhoods` restricted to neighborhoods that
    /// may have *grown* — i.e. those affected by `AddEdges` ops in this
    /// mutation. Used as the candidate set for batch compaction. The
    /// returned Vec is the raw walk and may contain duplicates; callers
    /// fold into a set to dedup.
    pub fn expanded_neighborhoods(&self) -> Vec<(VectorId, usize)> {
        let mut out = Vec::new();
        for op in &self.ops {
            if let MutationOp::AddEdges {
                base,
                layer,
                neighbors,
                edge_type,
            } = op
            {
                if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                    out.push((*base, *layer));
                }
                if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                    for n in neighbors {
                        out.push((*n, *layer));
                    }
                }
            }
        }
        out
    }

    /// Neighborhoods touched by `AddEdges` OR `RemoveEdges` in this mutation.
    /// Used by the per-slot invalid-link prune step: any neighborhood whose
    /// edge set was modified is a candidate for stale-reference cleanup.
    /// The returned Vec is the raw walk and may contain duplicates; callers
    /// fold into a set to dedup.
    pub fn updated_neighborhoods(&self) -> Vec<(VectorId, usize)> {
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                MutationOp::AddEdges {
                    base,
                    layer,
                    neighbors,
                    edge_type,
                }
                | MutationOp::RemoveEdges {
                    base,
                    layer,
                    neighbors,
                    edge_type,
                } => {
                    if matches!(edge_type, EdgeType::Base | EdgeType::All) {
                        out.push((*base, *layer));
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for n in neighbors {
                            out.push((*n, *layer));
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }
}

/// Type of edges between `base` and the nodes listed in `neighbors` affected by
/// edge mutations.
///
/// - `Base`: affects nodes listed in `neighbors` found in `base`'s neighbor
///   list (forward edges from `base`).
/// - `Neighbors`: affects instances of `base` in each neighbor node's neighbor
///   list (back-edges into `base`).
/// - `All`: affects both of the above types of edges (symmetric edges).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Affects forward edges from the base node
    Base,

    /// Affects back edges into the base node
    Neighbors,

    /// Affects both forward edges from and back edges into the base node
    All,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateEntryPoint {
    /// Do not update entry points based on inserted vector.
    False,

    /// Append a new entry point to the current list.
    Append { layer: usize },
}

#[cfg(test)]
mod tests {
    use crate::hnsw::graph::layered_graph::{Layer, Tick};
    use iris_mpc_common::VectorId;

    /// Test helper: the neighbor list of `node`, owned, or `None` if absent.
    fn links(layer: &Layer, node: u32) -> Option<Vec<u32>> {
        layer.get_links(&node).map(|n| n.neighbors().to_vec())
    }

    /// Test helper: add `node` as an incoming edge into each target's list —
    /// the back-edge half of an insert.
    fn add_backlinks(layer: &mut Layer, node: u32, targets: &[u32], seq: u64) {
        for &t in targets {
            layer.edit_links(t, Tick::new(seq), |_old, nbrs| nbrs.push(node));
        }
    }

    /// Test helper: remove `neighbors` from `node`'s own list, unidirectionally.
    fn remove_links(layer: &mut Layer, node: u32, neighbors: &[u32], seq: u64) {
        layer.edit_links(node, Tick::new(seq), |_old, nbrs| {
            nbrs.retain(|x| !neighbors.contains(x))
        });
    }

    // ── InsertNode ────────────────────────────────────────────────────────────

    /// Links are set to exactly the neighbors provided — no more, no less.
    #[test]
    fn insert_node_sets_exact_links() {
        let mut layer = Layer::new();
        layer.insert_node(1, vec![2, 3, 4], 0);
        assert_eq!(links(&layer, 1), Some(vec![2, 3, 4]));
    }

    /// A second InsertNode on the same id replaces the link list entirely.
    #[test]
    fn insert_node_replaces_existing_links() {
        let mut layer = Layer::new();
        layer.insert_node(1, vec![2, 3], 0);
        layer.insert_node(1, vec![4, 5], 0);
        assert_eq!(links(&layer, 1), Some(vec![4, 5]));
    }

    // ── AddNeighbors (edit_links back-edge half) ──────────────────────────────

    /// The inserted id appears in every neighborhood node's link list.
    #[test]
    fn add_neighbors_inserts_id_into_existing_nodes() {
        let mut layer = Layer::new();
        layer.insert_node(10, vec![], 0);
        layer.insert_node(20, vec![], 0);
        add_backlinks(&mut layer, 5, &[10, 20], 1);
        assert_eq!(links(&layer, 10), Some(vec![5]));
        assert_eq!(links(&layer, 20), Some(vec![5]));
    }

    /// Repeated calls with the same id are idempotent — no duplicate links.
    #[test]
    fn add_neighbors_is_idempotent() {
        let mut layer = Layer::new();
        layer.insert_node(10, vec![], 0);
        add_backlinks(&mut layer, 5, &[10], 1);
        add_backlinks(&mut layer, 5, &[10], 2);
        assert_eq!(links(&layer, 10), Some(vec![5]));
    }

    /// Links remain sorted after multiple add_neighbor calls in arbitrary order.
    #[test]
    fn add_neighbors_maintains_sorted_order() {
        let mut layer = Layer::new();
        layer.insert_node(10, vec![], 0);
        add_backlinks(&mut layer, 7, &[10], 1);
        add_backlinks(&mut layer, 3, &[10], 2);
        add_backlinks(&mut layer, 5, &[10], 3);
        assert_eq!(links(&layer, 10), Some(vec![3, 5, 7]));
    }

    /// edit_links silently skips nodes that don't exist. No phantom entries.
    #[test]
    fn add_neighbors_skips_nonexistent_nodes() {
        let mut layer = Layer::new();
        add_backlinks(&mut layer, 1, &[99], 1); // node 99 was never inserted
        assert!(layer.get_links(&99).is_none());
    }

    // ── RemoveNeighbors (edit_links removal) ──────────────────────────────────

    /// Only the specified neighbors are removed; all others are preserved.
    #[test]
    fn remove_neighbors_removes_specified_only() {
        let mut layer = Layer::new();
        layer.insert_node(1, vec![2, 3, 4, 5], 0);
        remove_links(&mut layer, 1, &[2, 4], 1);
        assert_eq!(links(&layer, 1), Some(vec![3, 5]));
    }

    /// Removal is unidirectional: only the target node's list is modified.
    #[test]
    fn remove_neighbors_is_unidirectional() {
        let mut layer = Layer::new();
        layer.insert_node(1, vec![2, 3], 0);
        layer.insert_node(2, vec![1, 3], 0);
        remove_links(&mut layer, 1, &[2], 1);
        assert_eq!(links(&layer, 1), Some(vec![3]));
        assert_eq!(links(&layer, 2), Some(vec![1, 3]));
    }

    /// edit_links on a node that doesn't exist is a no-op, not a panic.
    #[test]
    fn remove_neighbors_on_nonexistent_node_is_noop() {
        let mut layer = Layer::new();
        remove_links(&mut layer, 99, &[1, 2], 1); // should not panic
    }

    // ── WAL replay sequences ──────────────────────────────────────────────────

    /// Replays a typical insert: InsertNode sets the new node's forward links,
    /// then back-edges wire it into existing nodes.
    #[test]
    fn wal_replay_insert_then_backlinks() {
        let mut layer = Layer::new();
        layer.insert_node(10, vec![20, 30], 0);
        layer.insert_node(20, vec![10, 30], 0);
        layer.insert_node(30, vec![10, 20], 0);

        // New node 40 inserted with forward links to 10 and 20
        layer.insert_node(40, vec![10, 20], 0);
        // Backlinks: 10 and 20 each gain 40 as a neighbor
        add_backlinks(&mut layer, 40, &[10, 20], 1);

        assert_eq!(links(&layer, 40), Some(vec![10, 20]));
        assert_eq!(links(&layer, 10), Some(vec![20, 30, 40]));
        assert_eq!(links(&layer, 20), Some(vec![10, 30, 40]));
        // 30 was not in the backlink set — untouched
        assert_eq!(links(&layer, 30), Some(vec![10, 20]));
    }

    /// Replays a full mutation group: insert + backlinks + compaction.
    #[test]
    fn wal_replay_insert_backlinks_then_compact() {
        let mut layer = Layer::new();
        layer.insert_node(1, vec![2, 3], 0);
        layer.insert_node(2, vec![1, 3], 0);
        layer.insert_node(3, vec![1, 2], 0);

        // Insert node 4 with forward links [1, 2]
        layer.insert_node(4, vec![1, 2], 0);
        // Backlinks into 1 and 2
        add_backlinks(&mut layer, 4, &[1, 2], 1);
        // Compaction: node 2 now exceeds link limit, prune neighbor 3
        remove_links(&mut layer, 2, &[3], 2);

        assert_eq!(links(&layer, 4), Some(vec![1, 2]));
        assert_eq!(links(&layer, 1), Some(vec![2, 3, 4]));
        assert_eq!(links(&layer, 2), Some(vec![1, 4]));
        // unidirectional pruning — 3 still links back to 2
        assert_eq!(links(&layer, 3), Some(vec![1, 2]));
    }

    // ── expanded_neighborhoods ────────────────────────────────────────────────

    use super::{EdgeType, MutationOp, UnstampedMutation, UpdateEntryPoint};

    #[allow(dead_code)]
    fn mk_vector_id(id: u32) -> VectorId {
        VectorId::from_serial_id(id)
    }

    fn mk_add_edges(
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
        edge_type: EdgeType,
    ) -> MutationOp {
        MutationOp::AddEdges {
            base: mk_vector_id(base),
            neighbors: neighbors.into_iter().map(mk_vector_id).collect(),
            layer,
            edge_type,
        }
    }

    #[test]
    fn expanded_neighborhoods_addedges_all_yields_base_and_neighbors() {
        let mutation = UnstampedMutation {
            ops: vec![mk_add_edges(1, vec![2, 3], 0, EdgeType::All)],
        };
        let mut got = mutation.expanded_neighborhoods();
        got.sort();
        assert_eq!(
            got,
            vec![
                (mk_vector_id(1), 0),
                (mk_vector_id(2), 0),
                (mk_vector_id(3), 0)
            ]
        );
    }

    #[test]
    fn expanded_neighborhoods_addedges_base_yields_only_base() {
        let mutation = UnstampedMutation {
            ops: vec![mk_add_edges(1, vec![2, 3], 1, EdgeType::Base)],
        };
        assert_eq!(
            mutation.expanded_neighborhoods(),
            vec![(mk_vector_id(1), 1)]
        );
    }

    #[test]
    fn expanded_neighborhoods_addedges_neighbors_yields_only_neighbors() {
        let mutation = UnstampedMutation {
            ops: vec![mk_add_edges(1, vec![2, 3], 2, EdgeType::Neighbors)],
        };
        let mut got = mutation.expanded_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(mk_vector_id(2), 2), (mk_vector_id(3), 2)]);
    }

    #[test]
    fn expanded_neighborhoods_ignores_non_addedges_ops() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::AddNode {
                    id: mk_vector_id(1),
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::RemoveNode {
                    id: mk_vector_id(2),
                },
                MutationOp::RemoveEdges {
                    base: mk_vector_id(3),
                    neighbors: vec![mk_vector_id(4), mk_vector_id(5)],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                mk_add_edges(6, vec![7], 0, EdgeType::All),
            ],
        };
        let mut got = mutation.expanded_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(mk_vector_id(6), 0), (mk_vector_id(7), 0)]);
    }

    // ── updated_neighborhoods ─────────────────────────────────────────────────

    #[test]
    fn updated_neighborhoods_includes_removeedges() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::RemoveEdges {
                    base: mk_vector_id(1),
                    neighbors: vec![mk_vector_id(2), mk_vector_id(3)],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                MutationOp::AddEdges {
                    base: mk_vector_id(10),
                    neighbors: vec![mk_vector_id(11)],
                    layer: 1,
                    edge_type: EdgeType::Base,
                },
            ],
        };
        let mut got = mutation.updated_neighborhoods();
        got.sort();
        assert_eq!(
            got,
            vec![
                (mk_vector_id(1), 0),
                (mk_vector_id(2), 0),
                (mk_vector_id(3), 0),
                (mk_vector_id(10), 1)
            ]
        );
    }

    #[test]
    fn updated_neighborhoods_removeedges_respects_edge_type() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::RemoveEdges {
                    base: mk_vector_id(1),
                    neighbors: vec![mk_vector_id(2), mk_vector_id(3)],
                    layer: 0,
                    edge_type: EdgeType::Base,
                },
                MutationOp::RemoveEdges {
                    base: mk_vector_id(5),
                    neighbors: vec![mk_vector_id(6), mk_vector_id(7)],
                    layer: 0,
                    edge_type: EdgeType::Neighbors,
                },
            ],
        };
        let mut got = mutation.updated_neighborhoods();
        got.sort();
        assert_eq!(
            got,
            vec![
                (mk_vector_id(1), 0),
                (mk_vector_id(6), 0),
                (mk_vector_id(7), 0)
            ]
        );
    }

    #[test]
    fn updated_neighborhoods_ignores_addnode_removenode() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::AddNode {
                    id: mk_vector_id(1),
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::RemoveNode {
                    id: mk_vector_id(2),
                },
            ],
        };
        assert!(mutation.updated_neighborhoods().is_empty());
    }
}
