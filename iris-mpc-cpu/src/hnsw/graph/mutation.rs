use serde::{Deserialize, Serialize};

/// Represents a diff to apply to an existing graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphMutation<Vector: Ord> {
    // Removes the node's links from the graph and updates all its neighbors' neighborhoods,
    // tombstoning the iris code of the removed Vector.
    RemoveNode {
        id: Vector,
    },
    // Add bidirectional links.
    InsertNode {
        // List of layer, neighbors.
        layers: Vec<(usize, Vec<Vector>)>,
        update_ep: UpdateEntryPoint,
        id: Vector,
    },
    // Compact a neighborhood.
    Compact {
        // List of layer, neighbors.
        to_remove: Vec<(usize, Vec<Vector>)>,
        id: Vector,
    },
    // Overwrite a neighborhood. Used by re-randomization.
    Overwrite {
        // List of layer, neighbors.
        layers: Vec<(usize, Vec<Vector>)>,
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
