use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphMutation<V: Ord> {
    pub seq_no: u64,
    pub ops: Vec<MutationOp<V>>,
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
pub struct UnstampedMutation<V: Ord> {
    pub ops: Vec<MutationOp<V>>,
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
pub enum MutationOp<Vector: Ord> {
    AddNode {
        id: Vector,
        /// Number of real graph layers this node is included in. The node will
        /// be present in layers `0..height`.
        height: usize,
        update_ep: UpdateEntryPoint,
    },
    RemoveNode {
        id: Vector,
    },
    AddEdges {
        base: Vector,
        neighbors: Vec<Vector>,
        layer: usize,
        edge_type: EdgeType,
    },
    RemoveEdges {
        base: Vector,
        neighbors: Vec<Vector>,
        layer: usize,
        edge_type: EdgeType,
    },
}

impl<V: std::fmt::Debug + Ord> std::fmt::Debug for MutationOp<V> {
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

impl<V: Clone + Ord> UnstampedMutation<V> {
    /// Subset of `updated_neighborhoods` restricted to neighborhoods that
    /// may have *grown* — i.e. those affected by `AddEdges` ops in this
    /// mutation. Used as the candidate set for batch compaction. The
    /// returned Vec is the raw walk and may contain duplicates; callers
    /// fold into a set to dedup.
    pub fn expanded_neighborhoods(&self) -> Vec<(V, usize)> {
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
                    out.push((base.clone(), *layer));
                }
                if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                    for n in neighbors {
                        out.push((n.clone(), *layer));
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
    pub fn updated_neighborhoods(&self) -> Vec<(V, usize)> {
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
                        out.push((base.clone(), *layer));
                    }
                    if matches!(edge_type, EdgeType::Neighbors | EdgeType::All) {
                        for n in neighbors {
                            out.push((n.clone(), *layer));
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

    /// Set a new unique entry point.
    SetUnique { layer: usize },

    /// Append a new entry point to the current list.
    Append { layer: usize },
}

#[cfg(test)]
mod tests {
    use crate::hnsw::graph::layered_graph::Layer;

    // ── InsertNode ────────────────────────────────────────────────────────────

    /// Links are set to exactly the neighbors provided — no more, no less.
    #[test]
    fn insert_node_sets_exact_links() {
        let mut layer = Layer::new();
        layer.insert_node(&1i32, vec![2, 3, 4]);
        assert_eq!(layer.get_links(&1), Some([2, 3, 4].as_slice()));
    }

    /// A second InsertNode on the same id replaces the link list entirely.
    #[test]
    fn insert_node_replaces_existing_links() {
        let mut layer = Layer::new();
        layer.insert_node(&1i32, vec![2, 3]);
        layer.insert_node(&1i32, vec![4, 5]);
        assert_eq!(layer.get_links(&1), Some([4, 5].as_slice()));
    }

    // ── AddNeighbors ─────────────────────────────────────────────────────────

    /// The inserted id appears in every neighborhood node's link list.
    #[test]
    fn add_neighbors_inserts_id_into_existing_nodes() {
        let mut layer = Layer::new();
        layer.insert_node(&10i32, vec![]);
        layer.insert_node(&20i32, vec![]);
        layer.link_node_to_neighbors(&5, vec![10, 20]);
        assert_eq!(layer.get_links(&10), Some([5].as_slice()));
        assert_eq!(layer.get_links(&20), Some([5].as_slice()));
    }

    /// Repeated calls with the same id are idempotent — no duplicate links.
    /// This is the dedup contract that WAL replay depends on.
    #[test]
    fn add_neighbors_is_idempotent() {
        let mut layer = Layer::new();
        layer.insert_node(&10i32, vec![]);
        layer.link_node_to_neighbors(&5, vec![10]);
        layer.link_node_to_neighbors(&5, vec![10]);
        assert_eq!(layer.get_links(&10), Some([5].as_slice()));
    }

    /// Links remain sorted after multiple add_neighbor calls in arbitrary order.
    /// This ordering is part of the WAL replay contract — changing it silently
    /// breaks any stored WAL entry that relied on it.
    #[test]
    fn add_neighbors_maintains_sorted_order() {
        let mut layer = Layer::new();
        layer.insert_node(&10i32, vec![]);
        layer.link_node_to_neighbors(&7, vec![10]);
        layer.link_node_to_neighbors(&3, vec![10]);
        layer.link_node_to_neighbors(&5, vec![10]);
        assert_eq!(layer.get_links(&10), Some([3, 5, 7].as_slice()));
    }

    /// add_neighbor silently skips neighborhood nodes that don't exist.
    /// No phantom entries are created.
    #[test]
    fn add_neighbors_skips_nonexistent_nodes() {
        let mut layer = Layer::new();
        layer.link_node_to_neighbors(&1i32, vec![99]); // node 99 was never inserted
        assert!(layer.get_links(&99).is_none());
    }

    // ── RemoveNeighbors ───────────────────────────────────────────────────────

    /// Only the specified neighbors are removed; all others are preserved.
    #[test]
    fn remove_neighbors_removes_specified_only() {
        let mut layer = Layer::new();
        layer.insert_node(&1i32, vec![2, 3, 4, 5]);
        layer.unlink_neighbors_from_node(&1, vec![2, 4]);
        assert_eq!(layer.get_links(&1), Some([3, 5].as_slice()));
    }

    /// Removal is unidirectional: only the target node's list is modified.
    /// The removed neighbors' own link lists are untouched. This is the
    /// compaction contract — WAL replay must not infer bidirectional pruning.
    #[test]
    fn remove_neighbors_is_unidirectional() {
        let mut layer = Layer::new();
        layer.insert_node(&1i32, vec![2, 3]);
        layer.insert_node(&2i32, vec![1, 3]);
        layer.unlink_neighbors_from_node(&1, vec![2]);
        assert_eq!(layer.get_links(&1), Some([3].as_slice()));
        assert_eq!(layer.get_links(&2), Some([1, 3].as_slice()));
    }

    /// remove_neighbors on a node that doesn't exist is a no-op, not a panic.
    #[test]
    fn remove_neighbors_on_nonexistent_node_is_noop() {
        let mut layer = Layer::new();
        layer.unlink_neighbors_from_node(&99i32, vec![1, 2]); // should not panic
    }

    // ── WAL replay sequences ──────────────────────────────────────────────────

    /// Replays a typical insert: InsertNode sets the new node's forward links,
    /// then AddNeighbors wires the backlinks into existing nodes.
    /// The two operations are independent — InsertNode does not touch existing
    /// nodes, and AddNeighbors does not touch the inserted node's own list.
    #[test]
    fn wal_replay_insert_then_backlinks() {
        let mut layer = Layer::new();
        layer.insert_node(&10i32, vec![20, 30]);
        layer.insert_node(&20i32, vec![10, 30]);
        layer.insert_node(&30i32, vec![10, 20]);

        // New node 40 inserted with forward links to 10 and 20
        layer.insert_node(&40, vec![10, 20]);
        // Backlinks: 10 and 20 each gain 40 as a neighbor
        layer.link_node_to_neighbors(&40, vec![10, 20]);

        assert_eq!(layer.get_links(&40), Some([10, 20].as_slice()));
        assert_eq!(layer.get_links(&10).unwrap(), &[20, 30, 40]);
        assert_eq!(layer.get_links(&20).unwrap(), &[10, 30, 40]);
        // 30 was not in the backlink set — untouched
        assert_eq!(layer.get_links(&30).unwrap(), &[10, 20]);
    }

    /// Replays a full mutation group: insert + backlinks + compaction (RemoveNeighbors).
    /// Verifies the final state matches what would be computed at write time.
    #[test]
    fn wal_replay_insert_backlinks_then_compact() {
        let mut layer = Layer::new();
        layer.insert_node(&1i32, vec![2, 3]);
        layer.insert_node(&2i32, vec![1, 3]);
        layer.insert_node(&3i32, vec![1, 2]);

        // Insert node 4 with forward links [1, 2]
        layer.insert_node(&4, vec![1, 2]);
        // Backlinks into 1 and 2
        layer.link_node_to_neighbors(&4, vec![1, 2]);
        // Compaction: node 2 now exceeds link limit, prune neighbor 3
        layer.unlink_neighbors_from_node(&2, vec![3]);

        assert_eq!(layer.get_links(&4), Some([1, 2].as_slice()));
        assert_eq!(layer.get_links(&1).unwrap(), &[2, 3, 4]);
        assert_eq!(layer.get_links(&2).unwrap(), &[1, 4]);
        // unidirectional pruning — 3 still links back to 2
        assert_eq!(layer.get_links(&3).unwrap(), &[1, 2]);
    }

    // ── expanded_neighborhoods ────────────────────────────────────────────────

    use super::{EdgeType, MutationOp, UnstampedMutation, UpdateEntryPoint};

    fn mk_add_edges(
        base: i32,
        neighbors: Vec<i32>,
        layer: usize,
        edge_type: EdgeType,
    ) -> MutationOp<i32> {
        MutationOp::AddEdges {
            base,
            neighbors,
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
        assert_eq!(got, vec![(1, 0), (2, 0), (3, 0)]);
    }

    #[test]
    fn expanded_neighborhoods_addedges_base_yields_only_base() {
        let mutation = UnstampedMutation {
            ops: vec![mk_add_edges(1, vec![2, 3], 1, EdgeType::Base)],
        };
        assert_eq!(mutation.expanded_neighborhoods(), vec![(1, 1)]);
    }

    #[test]
    fn expanded_neighborhoods_addedges_neighbors_yields_only_neighbors() {
        let mutation = UnstampedMutation {
            ops: vec![mk_add_edges(1, vec![2, 3], 2, EdgeType::Neighbors)],
        };
        let mut got = mutation.expanded_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(2, 2), (3, 2)]);
    }

    #[test]
    fn expanded_neighborhoods_ignores_non_addedges_ops() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::AddNode {
                    id: 1,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::RemoveNode { id: 2 },
                MutationOp::RemoveEdges {
                    base: 3,
                    neighbors: vec![4, 5],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                mk_add_edges(6, vec![7], 0, EdgeType::All),
            ],
        };
        let mut got = mutation.expanded_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(6, 0), (7, 0)]);
    }

    // ── updated_neighborhoods ─────────────────────────────────────────────────

    #[test]
    fn updated_neighborhoods_includes_removeedges() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::RemoveEdges {
                    base: 1,
                    neighbors: vec![2, 3],
                    layer: 0,
                    edge_type: EdgeType::All,
                },
                MutationOp::AddEdges {
                    base: 10,
                    neighbors: vec![11],
                    layer: 1,
                    edge_type: EdgeType::Base,
                },
            ],
        };
        let mut got = mutation.updated_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(1, 0), (2, 0), (3, 0), (10, 1)]);
    }

    #[test]
    fn updated_neighborhoods_removeedges_respects_edge_type() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::RemoveEdges {
                    base: 1,
                    neighbors: vec![2, 3],
                    layer: 0,
                    edge_type: EdgeType::Base,
                },
                MutationOp::RemoveEdges {
                    base: 5,
                    neighbors: vec![6, 7],
                    layer: 0,
                    edge_type: EdgeType::Neighbors,
                },
            ],
        };
        let mut got = mutation.updated_neighborhoods();
        got.sort();
        assert_eq!(got, vec![(1, 0), (6, 0), (7, 0)]);
    }

    #[test]
    fn updated_neighborhoods_ignores_addnode_removenode() {
        let mutation = UnstampedMutation {
            ops: vec![
                MutationOp::AddNode {
                    id: 1,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::RemoveNode { id: 2 },
            ],
        };
        assert!(mutation.updated_neighborhoods().is_empty());
    }
}
