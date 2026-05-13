use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupedMutations<V: Ord>(pub Vec<GraphMutation<V>>);

// NOTE: if a new version of any mutation is needed (ex: InsertNodeV2) such that
// the new variant would behave differently than before and it is desired to still process
// old variants apppropriately, simply add the new variant to the END of GraphMutation. If
// the new variant is added to the end then bincode can deserialize it correctly. Adding a new variant
// in between existing ones will cause bincode to deserialize the old version of GraphMutation with garbage
// data for its fields.
//
/// Represents a diff to apply to an existing graph.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutation<Vector: Ord> {
    AddNode {
        // List of layer, neighbors.
        layers: Vec<(usize, Vec<Vector>)>,
        update_ep: UpdateEntryPoint,
        id: Vector,
    },
    RemoveNode {
        id: Vector,
    },
    AddEdges {
        id: Vector,
        layer: usize,
        to_add: Vec<Vector>,
        direction: EdgeDirection,
    },
    RemoveEdges {
        id: Vector,
        layer: usize,
        to_remove: Vec<Vector>,
        direction: EdgeDirection,
    },
}

impl<V: std::fmt::Debug + Ord> std::fmt::Debug for GraphMutation<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RemoveNode { id } => f.debug_struct("RemoveNode").field("id", id).finish(),
            Self::AddNode {
                layers: _,
                update_ep,
                id,
            } => f
                .debug_struct("InsertNode")
                .field("update_ep", update_ep)
                .field("id", id)
                .finish(),
            Self::AddEdges {
                id,
                layer,
                direction,
                ..
            } => f
                .debug_struct("AddEdges")
                .field("id", id)
                .field("layer", layer)
                .field("direction", direction)
                .finish(),
            Self::RemoveEdges {
                id,
                layer,
                direction,
                ..
            } => f
                .debug_struct("RemoveEdges")
                .field("id", id)
                .field("layer", layer)
                .field("direction", direction)
                .finish(),
        }
    }
}

/// Direction in which edge mutations are applied between `id` and the
/// vectors listed in `to_add` / `to_remove`.
///
/// - `Outgoing`: writes to `id`'s own neighbor list (forward edges from `id`).
/// - `Incoming`: writes `id` into each target's neighbor list (back-edges into `id`).
/// - `Bidirectional`: both of the above, applied together.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeDirection {
    Outgoing,
    Incoming,
    Bidirectional,
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
        layer.insert_node(1i32, vec![2, 3, 4]);
        assert_eq!(layer.get_links(&1), Some([2, 3, 4].as_slice()));
    }

    /// A second InsertNode on the same id replaces the link list entirely.
    #[test]
    fn insert_node_replaces_existing_links() {
        let mut layer = Layer::new();
        layer.insert_node(1i32, vec![2, 3]);
        layer.insert_node(1i32, vec![4, 5]);
        assert_eq!(layer.get_links(&1), Some([4, 5].as_slice()));
    }

    // ── AddNeighbors ─────────────────────────────────────────────────────────

    /// The inserted id appears in every neighborhood node's link list.
    #[test]
    fn add_neighbors_inserts_id_into_existing_nodes() {
        let mut layer = Layer::new();
        layer.insert_node(10i32, vec![]);
        layer.insert_node(20i32, vec![]);
        layer.add_neighbor(5, vec![10, 20]);
        assert_eq!(layer.get_links(&10), Some([5].as_slice()));
        assert_eq!(layer.get_links(&20), Some([5].as_slice()));
    }

    /// Repeated calls with the same id are idempotent — no duplicate links.
    /// This is the dedup contract that WAL replay depends on.
    #[test]
    fn add_neighbors_is_idempotent() {
        let mut layer = Layer::new();
        layer.insert_node(10i32, vec![]);
        layer.add_neighbor(5, vec![10]);
        layer.add_neighbor(5, vec![10]);
        assert_eq!(layer.get_links(&10), Some([5].as_slice()));
    }

    /// Links remain sorted after multiple add_neighbor calls in arbitrary order.
    /// This ordering is part of the WAL replay contract — changing it silently
    /// breaks any stored WAL entry that relied on it.
    #[test]
    fn add_neighbors_maintains_sorted_order() {
        let mut layer = Layer::new();
        layer.insert_node(10i32, vec![]);
        layer.add_neighbor(7, vec![10]);
        layer.add_neighbor(3, vec![10]);
        layer.add_neighbor(5, vec![10]);
        assert_eq!(layer.get_links(&10), Some([3, 5, 7].as_slice()));
    }

    /// add_neighbor silently skips neighborhood nodes that don't exist.
    /// No phantom entries are created.
    #[test]
    fn add_neighbors_skips_nonexistent_nodes() {
        let mut layer = Layer::new();
        layer.add_neighbor(1i32, vec![99]); // node 99 was never inserted
        assert!(layer.get_links(&99).is_none());
    }

    // ── RemoveNeighbors ───────────────────────────────────────────────────────

    /// Only the specified neighbors are removed; all others are preserved.
    #[test]
    fn remove_neighbors_removes_specified_only() {
        let mut layer = Layer::new();
        layer.insert_node(1i32, vec![2, 3, 4, 5]);
        layer.remove_neighbors(&1, vec![2, 4]);
        assert_eq!(layer.get_links(&1), Some([3, 5].as_slice()));
    }

    /// Removal is unidirectional: only the target node's list is modified.
    /// The removed neighbors' own link lists are untouched. This is the
    /// compaction contract — WAL replay must not infer bidirectional pruning.
    #[test]
    fn remove_neighbors_is_unidirectional() {
        let mut layer = Layer::new();
        layer.insert_node(1i32, vec![2, 3]);
        layer.insert_node(2i32, vec![1, 3]);
        layer.remove_neighbors(&1, vec![2]);
        assert_eq!(layer.get_links(&1), Some([3].as_slice()));
        assert_eq!(layer.get_links(&2), Some([1, 3].as_slice()));
    }

    /// remove_neighbors on a node that doesn't exist is a no-op, not a panic.
    #[test]
    fn remove_neighbors_on_nonexistent_node_is_noop() {
        let mut layer = Layer::new();
        layer.remove_neighbors(&99i32, vec![1, 2]); // should not panic
    }

    // ── WAL replay sequences ──────────────────────────────────────────────────

    /// Replays a typical insert: InsertNode sets the new node's forward links,
    /// then AddNeighbors wires the backlinks into existing nodes.
    /// The two operations are independent — InsertNode does not touch existing
    /// nodes, and AddNeighbors does not touch the inserted node's own list.
    #[test]
    fn wal_replay_insert_then_backlinks() {
        let mut layer = Layer::new();
        layer.insert_node(10i32, vec![20, 30]);
        layer.insert_node(20i32, vec![10, 30]);
        layer.insert_node(30i32, vec![10, 20]);

        // New node 40 inserted with forward links to 10 and 20
        layer.insert_node(40, vec![10, 20]);
        // Backlinks: 10 and 20 each gain 40 as a neighbor
        layer.add_neighbor(40, vec![10, 20]);

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
        layer.insert_node(1i32, vec![2, 3]);
        layer.insert_node(2i32, vec![1, 3]);
        layer.insert_node(3i32, vec![1, 2]);

        // Insert node 4 with forward links [1, 2]
        layer.insert_node(4, vec![1, 2]);
        // Backlinks into 1 and 2
        layer.add_neighbor(4, vec![1, 2]);
        // Compaction: node 2 now exceeds link limit, prune neighbor 3
        layer.remove_neighbors(&2, vec![3]);

        assert_eq!(layer.get_links(&4), Some([1, 2].as_slice()));
        assert_eq!(layer.get_links(&1).unwrap(), &[2, 3, 4]);
        assert_eq!(layer.get_links(&2).unwrap(), &[1, 4]);
        // unidirectional pruning — 3 still links back to 2
        assert_eq!(layer.get_links(&3).unwrap(), &[1, 2]);
    }
}
