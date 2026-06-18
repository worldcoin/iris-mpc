use std::{
    collections::HashSet,
    fmt::{Display, Formatter, Result},
};

use clap::ValueEnum;
use iris_mpc_common::VectorId;

use super::{jaccard::JaccardState, Differ};

#[derive(Debug, Clone, ValueEnum)]
pub enum SortBy {
    /// Sorts nodes by their natural order (e.g., index).
    Index,
    /// Sorts nodes by Jaccard similarity, from least similar to most similar.
    Jaccard,
}

impl Display for SortBy {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let s = match self {
            SortBy::Index => "Index",
            SortBy::Jaccard => "Jaccard",
        };
        write!(f, "{}", s)
    }
}

/// Contains the explicit differences between two neighborhoods for a single node.
#[derive(Debug, Clone)]
pub struct NeighborhoodDiff {
    pub only_in_lhs: HashSet<VectorId>,
    pub only_in_rhs: HashSet<VectorId>,
    pub jaccard_state: JaccardState,
}

#[derive(Debug, Clone)]
pub struct ExplicitDiff(pub Vec<LayerDiffResult>);

#[derive(Debug, Clone)]
pub struct LayerDiffResult {
    pub layer_index: usize,
    pub diffs: Vec<(VectorId, NeighborhoodDiff)>,
}

impl Display for ExplicitDiff {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        for layer_result in &self.0 {
            let nonempty_diffs: Vec<_> = layer_result
                .diffs
                .iter()
                .filter(|(_, d)| !d.only_in_lhs.is_empty() || !d.only_in_rhs.is_empty())
                .collect();
            writeln!(f, "--- Layer {} ---", layer_result.layer_index)?;
            if nonempty_diffs.is_empty() {
                writeln!(f, "No differences found.")?;
            } else {
                for (node, diff) in nonempty_diffs {
                    writeln!(
                        f,
                        "Node: {} (Jaccard: {:.4})",
                        node,
                        diff.jaccard_state.compute()
                    )?;
                    #[allow(
                        clippy::iter_over_hash_type,
                        reason = "Only used for display, not relying on order."
                    )]
                    for neighbor in &diff.only_in_lhs {
                        writeln!(f, "  + {}", neighbor)?;
                    }
                    #[allow(
                        clippy::iter_over_hash_type,
                        reason = "Only used for display, not relying on order."
                    )]
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
pub struct ExplicitNeighborhoodDiffer {
    sort_by: SortBy,
    per_layer_results: Vec<LayerDiffResult>,
    current_layer_diffs: Vec<(VectorId, NeighborhoodDiff)>,
}

impl ExplicitNeighborhoodDiffer {
    pub fn new(sort_by: SortBy) -> Self {
        Self {
            sort_by,
            per_layer_results: Vec::new(),
            current_layer_diffs: Vec::new(),
        }
    }
}

impl Differ for ExplicitNeighborhoodDiffer {
    type Output = ExplicitDiff;

    fn start_layer(&mut self, _layer_index: usize) {
        self.current_layer_diffs.clear();
    }

    fn diff_neighborhood(
        &mut self,
        _layer_index: usize,
        node: &VectorId,
        lhs: &[VectorId],
        rhs: &[VectorId],
    ) {
        let lhs_set: HashSet<_> = lhs.iter().cloned().collect();
        let rhs_set: HashSet<_> = rhs.iter().cloned().collect();

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

        self.current_layer_diffs.push((*node, diff));
    }

    fn end_layer(&mut self, layer_index: usize) {
        if !self.current_layer_diffs.is_empty() {
            match self.sort_by {
                SortBy::Jaccard => {
                    self.current_layer_diffs
                        .sort_by(|a, b| a.1.jaccard_state.compare_as_fractions(&b.1.jaccard_state));
                }
                SortBy::Index => {
                    self.current_layer_diffs.sort_by_key(|a| a.0);
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
