use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

use crate::hnsw::{
    graph::{layered_graph::Layer, neighborhood::SortedEdgeIds},
    vector_store::Ref,
    GraphMem,
};

pub mod combinators;
pub mod jaccard;
pub mod node_equiv;

pub trait NeighborhoodDiffer<V> {
    type NeighborhoodDiff: Debug + Clone;

    fn diff_neighborhood(
        &self,
        lhs: &SortedEdgeIds<V>,
        rhs: &SortedEdgeIds<V>,
    ) -> Self::NeighborhoodDiff;
}

pub trait LayerDiffer<V: Ref + Display + FromStr> {
    type LayerDiff: Debug + Clone;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff;
}

pub trait GraphDiffer<V: Ref + Display + FromStr> {
    type GraphDiff: Debug + Clone;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff;
}
