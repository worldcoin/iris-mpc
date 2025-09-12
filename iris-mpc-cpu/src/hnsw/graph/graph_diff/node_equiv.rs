use std::collections::HashSet;

use crate::hnsw::graph::graph_diff::combinators::PerLayerCollector;

use super::*;

/// Diffs node and layer structure, ensuring that two graphs have the same number of layers
/// and the same (unordered) set of nodes in each layer. If this property does not hold, it returns a reason.
#[derive(Default)]
pub struct NodeEquivalence;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntraLayerResult<V> {
    OnlyInLhs(V),
    OnlyInRhs(V),
    None,
}

impl<V: Ref + Display + FromStr + Clone + Eq + std::hash::Hash> LayerDiffer<V> for NodeEquivalence {
    type LayerDiff = IntraLayerResult<V>;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        let lhs_keys: HashSet<_> = lhs.links.keys().cloned().collect();
        let rhs_keys: HashSet<_> = rhs.links.keys().cloned().collect();

        for v in lhs_keys.difference(&rhs_keys) {
            return IntraLayerResult::OnlyInLhs(v.clone());
        }
        for v in rhs_keys.difference(&lhs_keys) {
            return IntraLayerResult::OnlyInRhs(v.clone());
        }
        IntraLayerResult::None
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeEquivResult<V> {
    LayerCountMismatch {
        lhs_count: usize,
        rhs_count: usize,
    },
    LayerVertexDiff {
        layer_index: usize,
        diff: IntraLayerResult<V>,
    },
    Equivalent,
}

impl<V: Ref + Display + FromStr> GraphDiffer<V> for NodeEquivalence {
    type GraphDiff = NodeEquivResult<V>;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        if lhs.layers.len() != rhs.layers.len() {
            return NodeEquivResult::LayerCountMismatch {
                lhs_count: lhs.layers.len(),
                rhs_count: rhs.layers.len(),
            };
        }
        for (i, layer_diff) in PerLayerCollector(NodeEquivalence)
            .diff_graph(lhs, rhs)
            .into_iter()
            .enumerate()
        {
            if layer_diff != IntraLayerResult::None {
                return NodeEquivResult::LayerVertexDiff {
                    layer_index: i,
                    diff: layer_diff,
                };
            }
        }
        NodeEquivResult::Equivalent
    }
}
