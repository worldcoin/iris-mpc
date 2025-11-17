use std::collections::HashSet;
use std::fmt::Display;
use std::str::FromStr;

use crate::hnsw::vector_store::Ref;
use crate::hnsw::GraphMem;

/// Describes how the node and layer structure of two graphs are not equivalent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeEquivalenceError<V> {
    /// The graphs have a different number of layers.
    LayerCountMismatch { lhs_count: usize, rhs_count: usize },
    /// A node was found in the left-hand graph's layer that was not in the right's.
    NodeMissingInRhs { layer_index: usize, node: V },
    /// A node was found in the right-hand graph's layer that was not in the left's.
    NodeMissingInLhs { layer_index: usize, node: V },
}

/// Diffs the node and layer structure of two graphs.
///
/// This function ensures that two graphs have the same number of layers and the
/// same (unordered) set of nodes in each corresponding layer.
///
/// # Returns
/// - `Ok(())` if the graphs are equivalent in their node and layer structure.
/// - `Err(NodeEquivalenceError)` if they are not, with a reason for the failure.
pub fn ensure_node_equivalence<V: Ref + Display + FromStr + std::cmp::Ord>(
    lhs: &GraphMem<V>,
    rhs: &GraphMem<V>,
) -> Result<(), NodeEquivalenceError<V>> {
    // First, check if the number of layers is the same.
    if lhs.layers.len() != rhs.layers.len() {
        return Err(NodeEquivalenceError::LayerCountMismatch {
            lhs_count: lhs.layers.len(),
            rhs_count: rhs.layers.len(),
        });
    }

    // Iterate over each pair of layers to compare their nodes.
    for (i, (lhs_layer, rhs_layer)) in lhs.layers.iter().zip(rhs.layers.iter()).enumerate() {
        let lhs_nodes: HashSet<_> = lhs_layer.links.keys().cloned().collect();
        let rhs_nodes: HashSet<_> = rhs_layer.links.keys().cloned().collect();

        // Check for any node that exists in the left layer but not the right.
        if let Some(missing_node) = lhs_nodes.difference(&rhs_nodes).next() {
            return Err(NodeEquivalenceError::NodeMissingInRhs {
                layer_index: i,
                node: missing_node.clone(),
            });
        }

        // Check for any node that exists in the right layer but not the left.
        if let Some(missing_node) = rhs_nodes.difference(&lhs_nodes).next() {
            return Err(NodeEquivalenceError::NodeMissingInLhs {
                layer_index: i,
                node: missing_node.clone(),
            });
        }
    }

    // If all checks pass, the graphs are equivalent.
    Ok(())
}
