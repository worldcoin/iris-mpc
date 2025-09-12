use super::*;
use crate::hnsw::graph::graph_diff::NeighborhoodDiffer;

/// Collects diffs for all nodes from a layer into a Vec, for a given node differ
pub struct IntraLayerProcessor<
    V: Ref + Display + FromStr,
    ND: NeighborhoodDiffer<V>,
    LDRET: Clone + Debug,
> {
    combinator: dyn Fn(Vec<(V, ND::NeighborhoodDiff)>) -> LDRET,
}

impl<V: Ref + Display + FromStr, ND: NeighborhoodDiffer<V> + Default, LDRET: Clone + Debug>
    LayerDiffer<V> for IntraLayerProcessor<V, ND, LDRET>
{
    type LayerDiff = LDRET;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        (self.combinator)(
            lhs.links
                .iter()
                .filter_map(|(v, ne)| {
                    rhs.links
                        .get(v)
                        .map(|ner| (v.clone(), ND::default().diff_neighborhood(ne, ner)))
                })
                .collect(),
        )
    }
}

/// Collects diffs for all layers from a graph into a Vec, for a given layer differ
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
