use std::fmt::Display;
use std::str::FromStr;

use crate::hnsw::{vector_store::Ref, GraphMem};

pub mod explicit;
pub mod jaccard;
pub mod node_equiv;

/// A trait that defines a graph diffing strategy using a visitor pattern.
///
/// A `Differ` implementation can maintain internal state and update it as the
/// `run_diff` function traverses the layers and nodes of the graphs.
pub trait Differ<V: Ref + Display + FromStr + Ord> {
    /// The final output type of the diffing operation.
    type Output: Display;

    /// Called once before graph traversal begins.
    fn start_graph(&mut self) {}

    /// Called before traversing each layer.
    fn start_layer(&mut self, _layer_index: usize) {}

    /// Called for each node that exists in both the `lhs` and `rhs` layer.
    fn diff_neighborhood(&mut self, layer_index: usize, node: &V, lhs: &[V], rhs: &[V]);

    /// Called after traversing each layer.
    fn end_layer(&mut self, _layer_index: usize) {}

    /// Called at the very end to consume the differ and produce the final result.
    fn finish(self) -> Self::Output;
}

/// Traverses two graphs and applies a `Differ` to compute a result.
///
/// It's recommended to run `ensure_node_equivalence` before using this function
/// to ensure the graphs have a comparable structure.
pub fn run_diff<V, D>(lhs: &GraphMem<V>, rhs: &GraphMem<V>, mut differ: D) -> D::Output
where
    V: Ref + Display + FromStr + Ord,
    D: Differ<V>,
{
    differ.start_graph();

    for (i, (lhs_layer, rhs_layer)) in lhs.layers.iter().zip(rhs.layers.iter()).enumerate() {
        differ.start_layer(i);
        #[allow(clippy::iter_over_hash_type, reason = "TODO")]
        for (node, lhs_nbhd) in lhs_layer.links.iter() {
            if let Some(rhs_nbhd) = rhs_layer.links.get(node) {
                differ.diff_neighborhood(i, node, lhs_nbhd, rhs_nbhd);
            }
        }
        differ.end_layer(i);
    }

    differ.finish()
}
