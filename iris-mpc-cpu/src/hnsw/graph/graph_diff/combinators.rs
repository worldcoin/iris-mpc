use super::*;
use crate::hnsw::graph::graph_diff::NeighborhoodDiffer;

/// Collects diffs for all nodes from a layer into a Vec, for a given node differ
#[derive(Default)]
pub struct PerNodeCollector<ND>(pub ND);

impl<ND: NeighborhoodDiffer<V>, V: Ref + Display + FromStr> LayerDiffer<V>
    for PerNodeCollector<ND>
{
    type LayerDiff = Vec<ND::NeighborhoodDiff>;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        let mut ret = Vec::with_capacity(lhs.links.len());
        for node_ref in lhs.links.keys() {
            if let (Some(lhsn), Some(rhsn)) = (lhs.links.get(node_ref), rhs.links.get(node_ref)) {
                ret.push(self.0.diff_neighborhood(lhsn, rhsn))
            }
        }
        ret
    }
}

/// Collects diffs for all layers in into a Vec, for a given layer differ
#[derive(Default)]
pub struct PerLayerCollector<LD>(pub LD);

impl<LD: LayerDiffer<V>, V: Ref + Display + FromStr> GraphDiffer<V> for PerLayerCollector<LD> {
    type GraphDiff = Vec<LD::LayerDiff>;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        lhs.layers
            .iter()
            .zip(rhs.layers.iter())
            .map(|(lhs_layer, rhs_layer)| self.0.diff_layer(lhs_layer, rhs_layer))
            .collect()
    }
}
