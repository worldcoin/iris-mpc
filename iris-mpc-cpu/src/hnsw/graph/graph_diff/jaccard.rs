use crate::hnsw::graph::graph_diff::combinators::{PerLayerCollector, PerNodeCollector};

use super::*;
use std::{collections::HashSet, ops::Add};

#[derive(Default)]
pub struct JaccardSimilarity;

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
impl<V: Ref + Display + FromStr> NeighborhoodDiffer<V> for JaccardSimilarity {
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
impl<V: Ref + Display + FromStr> LayerDiffer<V> for JaccardSimilarity {
    type LayerDiff = JaccardState;

    fn diff_layer(&self, lhs: &Layer<V>, rhs: &Layer<V>) -> Self::LayerDiff {
        let mut total = JaccardState::default();
        for per_node in PerNodeCollector(JaccardSimilarity).diff_layer(lhs, rhs) {
            total = total + per_node;
        }
        total
    }
}

// Aggregates JaccardStates to provide similarity for entire graphs
impl<V: Ref + Display + FromStr> GraphDiffer<V> for JaccardSimilarity {
    type GraphDiff = JaccardState;

    fn diff_graph(&self, lhs: &GraphMem<V>, rhs: &GraphMem<V>) -> Self::GraphDiff {
        let mut total = JaccardState::default();
        for per_layer in PerLayerCollector(JaccardSimilarity).diff_graph(lhs, rhs) {
            total = total + per_layer;
        }
        total
    }
}
