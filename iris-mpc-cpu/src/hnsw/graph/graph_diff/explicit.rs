use std::{
    collections::HashSet,
    fmt::{Display, Formatter, Result},
    str::FromStr,
};

use super::{jaccard::JaccardState, Differ};
use crate::hnsw::{graph::neighborhood::SortedEdgeIds, vector_store::Ref};

#[derive(Debug, Clone)]
pub enum SortBy {
    /// Sorts nodes by their natural order (e.g., index).
    Index,
    /// Sorts nodes by Jaccard similarity, from least similar to most similar.
    Jaccard,
}

/// Contains the explicit differences between two neighborhoods for a single node.
#[derive(Debug, Clone)]
pub struct NeighborhoodDiff<V: Ref> {
    pub only_in_lhs: HashSet<V>,
    pub only_in_rhs: HashSet<V>,
    pub jaccard_state: JaccardState,
}

#[derive(Debug, Clone)]
pub struct ExplicitDiff<V: Ref>(pub Vec<LayerDiffResult<V>>);

#[derive(Debug, Clone)]
pub struct LayerDiffResult<V: Ref> {
    pub layer_index: usize,
    pub diffs: Vec<(V, NeighborhoodDiff<V>)>,
}

impl<V: Ref + Display> Display for ExplicitDiff<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for layer_result in &self.0 {
            writeln!(f, "--- Layer {} ---", layer_result.layer_index)?;
            if layer_result.diffs.is_empty() {
                writeln!(f, "No differences found.")?;
            }
            for (node, diff) in &layer_result.diffs {
                // Only print nodes that actually have differences.
                if !diff.only_in_lhs.is_empty() || !diff.only_in_rhs.is_empty() {
                    writeln!(
                        f,
                        "Node: {} (Jaccard: {:.4})",
                        node,
                        diff.jaccard_state.compute()
                    )?;
                    for neighbor in &diff.only_in_lhs {
                        writeln!(f, "  + {}", neighbor)?;
                    }
                    for neighbor in &diff.only_in_rhs {
                        writeln!(f, "  - {}", neighbor)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A differ that returns the explicit lists of node differences between neighborhoods.
pub struct ExplicitNeighborhoodDiffer<V: Ref> {
    sort_by: SortBy,
    per_layer_results: Vec<LayerDiffResult<V>>,
    current_layer_diffs: Vec<(V, NeighborhoodDiff<V>)>,
}

impl<V: Ref> ExplicitNeighborhoodDiffer<V> {
    pub fn new(sort_by: SortBy) -> Self {
        Self {
            sort_by,
            per_layer_results: Vec::new(),
            current_layer_diffs: Vec::new(),
        }
    }
}

impl<V: Ref + Display + FromStr + Ord> Differ<V> for ExplicitNeighborhoodDiffer<V> {
    type Output = ExplicitDiff<V>;

    fn start_layer(&mut self, _layer_index: usize) {
        self.current_layer_diffs.clear();
    }

    fn diff_neighborhood(
        &mut self,
        _layer_index: usize,
        node: &V,
        lhs: &SortedEdgeIds<V>,
        rhs: &SortedEdgeIds<V>,
    ) {
        let lhs_set: HashSet<_> = lhs.0.iter().cloned().collect();
        let rhs_set: HashSet<_> = rhs.0.iter().cloned().collect();

        let only_in_lhs: HashSet<_> = lhs_set.difference(&rhs_set).cloned().collect();
        let only_in_rhs: HashSet<_> = rhs_set.difference(&lhs_set).cloned().collect();

        let intersection = lhs_set.intersection(&rhs_set).count();
        let union = lhs_set.union(&rhs_set).count();
        let jaccard_state = JaccardState::new(intersection, union);

        let diff = NeighborhoodDiff {
            only_in_lhs,
            only_in_rhs,
            jaccard_state,
        };

        self.current_layer_diffs.push((node.clone(), diff));
    }

    fn end_layer(&mut self, layer_index: usize) {
        if !self.current_layer_diffs.is_empty() {
            match self.sort_by {
                SortBy::Jaccard => {
                    self.current_layer_diffs
                        .sort_by(|a, b| a.1.jaccard_state.compare_as_fractions(&b.1.jaccard_state));
                }
                SortBy::Index => {
                    self.current_layer_diffs.sort_by(|a, b| a.0.cmp(&b.0));
                }
            }
            self.per_layer_results.push(LayerDiffResult {
                layer_index,
                diffs: std::mem::take(&mut self.current_layer_diffs),
            });
        }
    }

    fn finish(self) -> Self::Output {
        ExplicitDiff(self.per_layer_results)
    }
}
