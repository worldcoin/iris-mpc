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

pub trait NeighborhoodDiffer<V>
where
    Self: Default,
{
    type NeighborhoodDiff: Debug + Clone;

    fn diff_neighborhood(
        &self,
        lhs: &SortedEdgeIds<V>,
        rhs: &SortedEdgeIds<V>,
    ) -> Self::NeighborhoodDiff;
}

pub trait LayerDiffer<V: Ref + Display + FromStr>
where
    Self: Default,
{
    type LayerDiff: Debug + Clone;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff;
}

pub trait GraphDiffer<V: Ref + Display + FromStr>
where
    Self: Default,
{
    type GraphDiff: Debug + Clone;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff;
}

/// Compares node equivalence of two graphs and prints the result if negative
/// Otherwise, prints Jaccard similarity for the edges of the two graphs
pub fn edge_diff<V: Ref + Display + FromStr>(lhs: &GraphMem<V>, rhs: &GraphMem<V>) {
    match node_equiv::NodeEquivalence::default().diff_graph(lhs, rhs) {
        node_equiv::GraphResult::Equivalent => {
            let result = jaccard::DetailedJaccard { n: 20 }.diff_graph(lhs, rhs);
            dbg!(result);
        }
        non_equiv => {
            dbg!(&non_equiv);
        }
    }
}
