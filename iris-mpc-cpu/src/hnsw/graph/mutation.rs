use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupedMutations<V: Ord>(pub Vec<GraphMutation<V>>);

/// Represents a diff to apply to an existing graph.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutation<Vector: Ord> {
    RemoveNode {
        id: Vector,
    },
    // delete the old entries in the graph without removing from thee entrypoints
    ReplaceNode {
        id: Vector,
    },
    InsertNode {
        // List of layer, neighbors.
        layers: Vec<(usize, Vec<Vector>)>,
        update_ep: UpdateEntryPoint,
        id: Vector,
    },
    Compact {
        to_remove: Vec<Vector>,
        layer: usize,
        id: Vector,
    },
}

impl<V: std::fmt::Debug + Ord> std::fmt::Debug for GraphMutation<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RemoveNode { id } => f.debug_struct("RemoveNode").field("id", id).finish(),
            Self::ReplaceNode { id } => f.debug_struct("ReplaceNode").field("id", id).finish(),
            Self::InsertNode {
                layers: _,
                update_ep,
                id,
            } => f
                .debug_struct("InsertNode")
                .field("update_ep", update_ep)
                .field("id", id)
                .finish(),
            Self::Compact {
                to_remove: _,
                layer,
                id,
            } => f
                .debug_struct("Compact")
                // .field("to_remove", to_remove)
                .field("layer", layer)
                .field("id", id)
                .finish(),
        }
    }
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

impl<Vector: Ord> GraphMutation<Vector> {
    /// Returns the current version of the GraphMutation format.
    ///
    /// This should be incremented whenever the GraphMutation enum or its variants
    /// are modified in an incompatible way. Keeping this separate allows old
    /// mutations to remain stored and handled appropriately during format upgrades.
    pub fn get_version() -> i32 {
        1
    }
}
