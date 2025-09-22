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

    fn diff_neighborhood(lhs: &SortedEdgeIds<V>, rhs: &SortedEdgeIds<V>) -> Self::NeighborhoodDiff;
}

pub trait LayerDiffer<V: Ref + Display + FromStr> {
    type LayerDiff: Debug + Clone;
    type ND: NeighborhoodDiffer<V>;

    fn accumulate(
        &self,
        per_nb: Vec<(V, <Self::ND as NeighborhoodDiffer<V>>::NeighborhoodDiff)>,
    ) -> Self::LayerDiff;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        self.accumulate(
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

    fn accumulate(&self, per_l: Vec<<Self::LD as LayerDiffer<V>>::LayerDiff>) -> Self::GraphDiff;

    fn diff_graph(&self, ld: Self::LD, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        self.accumulate(
            lhs.layers
                .iter()
                .zip(rhs.layers.iter())
                .map(|(lhs_layer, rhs_layer)| ld.diff_layer(lhs_layer, rhs_layer))
                .collect(),
        )
    }
}
