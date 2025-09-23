use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

use crate::hnsw::{
    graph::{layered_graph::Layer, neighborhood::SortedEdgeIds},
    vector_store::Ref,
    GraphMem,
};

pub mod jaccard;
pub mod node_equiv;

pub trait NeighborhoodDiffer<V> {
    type NeighborhoodDiff: Debug + Clone;

    // User defined method to diff two neighborhoods
    fn diff_neighborhood(lhs: &SortedEdgeIds<V>, rhs: &SortedEdgeIds<V>) -> Self::NeighborhoodDiff;
}

pub trait LayerDiffer<V: Ref + Display + FromStr> {
    type LayerDiff: Debug + Clone;
    type ND: NeighborhoodDiffer<V>;

    /// User-defined method of combining neighborhood results into a layer result
    fn accumulate_neighborhoods(
        &self,
        per_nb: Vec<(V, <Self::ND as NeighborhoodDiffer<V>>::NeighborhoodDiff)>,
    ) -> Self::LayerDiff;

    /// Main method to diff two layers; defaults to calling `accumulate_neighborhoods`
    /// on all neighborhoods of nodes present in layer `lhs`.
    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        self.accumulate_neighborhoods(
            lhs.links
                .iter()
                .filter_map(|(v, ne)| {
                    rhs.links
                        .get(v)
                        .map(|ner| (v.clone(), Self::ND::diff_neighborhood(ne, ner)))
                })
                .collect(),
        )
    }
}

pub trait GraphDiffer<V: Ref + Display + FromStr> {
    type GraphDiff: Debug + Clone;
    type LD: LayerDiffer<V>;

    /// User-defined method of combining layer results into a graph result
    fn accumulate_layers(
        &self,
        per_l: Vec<<Self::LD as LayerDiffer<V>>::LayerDiff>,
    ) -> Self::GraphDiff;

    /// Main method to diff two graphs; defaults to calling `accumulate_layers`
    /// on all layers present in graph `lhs`.
    ///
    /// Note that this default may return unexpected results if the two graphs are not node-equivalent.
    /// To adress this issue, use the method in conjunction with `ensure_node_equivalence`.
    fn diff_graph(&self, ld: Self::LD, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        self.accumulate_layers(
            lhs.layers
                .iter()
                .zip(rhs.layers.iter())
                .map(|(lhs_layer, rhs_layer)| ld.diff_layer(lhs_layer, rhs_layer))
                .collect(),
        )
    }
}
