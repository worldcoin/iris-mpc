use serde::{Deserialize, Serialize};

/// Represents a diff to apply to an existing graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutation<Vector: Ord> {
    // Removes the node's links from the graph and updates all its neighbors' neighborhoods
    RemoveNode {
        id: Vector,
    },
    // Add bidirectional links.
    InsertNode {
        // List of layer, neighbors.
        layers: Vec<(u32, Vec<Vector>)>,
        update_ep: UpdateEntryPoint,
        id: Vector,
    },
    // Compact a neighborhood.
    Compact {
        to_remove: Vec<Vector>,
        layer: u32,
        id: Vector,
    },
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
