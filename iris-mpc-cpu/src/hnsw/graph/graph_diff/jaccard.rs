use itertools::Itertools;

use crate::hnsw::graph::graph_diff::combinators::{PerLayerCollector, PerNodeCollector};

use super::*;
use std::{collections::HashSet, ops::Add};

#[derive(Default)]
pub struct SimpleJaccard;

#[derive(Debug, Default, Clone)]
pub struct JaccardState {
    pub intersection: usize,
    pub union: usize,
}

impl JaccardState {
    pub fn new(intersection: usize, union: usize) -> Self {
        Self {
            intersection,
            union,
        }
    }
    /// Calculates the Jaccard similarity, a value between 0.0 and 1.0.
    pub fn compute(&self) -> f64 {
        if self.union == 0 {
            1.0
        } else {
            self.intersection as f64 / self.union as f64
        }
    }

    pub fn compare_as_fractions(&self, other: &Self) -> std::cmp::Ordering {
        // Compare a/b vs c/d as a*d vs c*b
        (self.intersection * other.union).cmp(&(other.intersection * self.union))
    }
}
impl Add for JaccardState {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            intersection: self.intersection + rhs.intersection,
            union: self.union + rhs.union,
        }
    }
}

// Simple Jaccard similarity (size of intersection / size of union)
impl<V: Ref + Display + FromStr> NeighborhoodDiffer<V> for SimpleJaccard {
    type NeighborhoodDiff = JaccardState;

    fn diff_neighborhood(
        &self,
        lhs: &SortedEdgeIds<V>,
        rhs: &SortedEdgeIds<V>,
    ) -> Self::NeighborhoodDiff {
        let lhs: HashSet<_> = lhs.0.iter().cloned().collect();
        let rhs: HashSet<_> = rhs.0.iter().cloned().collect();
        JaccardState::new(lhs.intersection(&rhs).count(), lhs.union(&rhs).count())
    }
}

// Aggregates JaccardStates to provide similarity for entire layers
impl<V: Ref + Display + FromStr> LayerDiffer<V> for SimpleJaccard {
    type LayerDiff = JaccardState;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        let mut total = JaccardState::default();
        for per_node in PerNodeCollector(SimpleJaccard).diff_layer(lhs, rhs) {
            total = total + per_node;
        }
        total
    }
}

// Aggregates JaccardStates to provide similarity for entire graphs
impl<V: Ref + Display + FromStr> GraphDiffer<V> for SimpleJaccard {
    type GraphDiff = JaccardState;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        let mut total = JaccardState::default();
        for per_layer in PerLayerCollector(SimpleJaccard).diff_graph(lhs, rhs) {
            total = total + per_layer;
        }
        total
    }
}

/// Jaccard similarity for edges of entire graph, individual layers
/// and `n` most disimilar nodes for each individual layer
#[derive(Clone, Default)]
pub struct DetailedJaccard {
    pub n: usize,
}

impl<V: Ref + Display + FromStr> LayerDiffer<V> for DetailedJaccard {
    type LayerDiff = (JaccardState, Vec<JaccardState>);

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        let most_disimilar = PerNodeCollector(SimpleJaccard)
            .diff_layer(lhs, rhs)
            .into_iter()
            .sorted_by(|lhs, rhs| lhs.compare_as_fractions(rhs))
            .take(self.n)
            .collect_vec();
        let agg = SimpleJaccard::default().diff_layer(lhs, rhs);
        (agg, most_disimilar)
    }
}

impl<V: Ref + Display + FromStr> GraphDiffer<V> for DetailedJaccard {
    type GraphDiff = (JaccardState, Vec<(JaccardState, Vec<JaccardState>)>);

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        let most_disimilar_per_layer = PerLayerCollector(self.clone()).diff_graph(lhs, rhs);
        let agg = SimpleJaccard::default().diff_graph(lhs, rhs);
        (agg, most_disimilar_per_layer)
    }
}
