use super::*;
use std::{collections::HashSet, ops::Add};

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

impl Display for JaccardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.4} ({}/{})",
            self.compute(),
            self.intersection,
            self.union
        )
    }
}

/// Computes Jaccard similarity for edges.
/// Aggregated over graphs, individual layers or individual nodes
pub struct JaccardND;

impl<V: Ref + Display + FromStr> NeighborhoodDiffer<V> for JaccardND {
    type NeighborhoodDiff = JaccardState;

    fn diff_neighborhood(lhs: &SortedEdgeIds<V>, rhs: &SortedEdgeIds<V>) -> Self::NeighborhoodDiff {
        let lhs: HashSet<_> = lhs.0.iter().cloned().collect();
        let rhs: HashSet<_> = rhs.0.iter().cloned().collect();
        JaccardState::new(lhs.intersection(&rhs).count(), lhs.union(&rhs).count())
    }
}

pub struct JaccardLD;

// Aggregates JaccardStates to compute similarity for entire layers
impl<V: Ref + Display + FromStr> LayerDiffer<V> for JaccardLD {
    type LayerDiff = JaccardState;
    type ND = JaccardND;
    fn accumulate(
        &self,
        per_nb: Vec<(V, <Self::ND as NeighborhoodDiffer<V>>::NeighborhoodDiff)>,
    ) -> Self::LayerDiff {
        let mut ret = JaccardState::default();
        for (_, jacc_nb) in per_nb.into_iter() {
            ret = ret + jacc_nb;
        }
        ret
    }
}

pub struct JaccardGD;

// Aggregates JaccardStates to compute similarity for entire graphs
impl<V: Ref + Display + FromStr> GraphDiffer<V> for JaccardGD {
    type GraphDiff = JaccardState;
    type LD = JaccardLD;

    fn accumulate(&self, per_nb: Vec<<Self::LD as LayerDiffer<V>>::LayerDiff>) -> Self::GraphDiff {
        let mut acc = JaccardState::default();
        for jacc_lay in per_nb.into_iter() {
            acc = acc + jacc_lay;
        }
        acc
    }
}

/// Jaccard similarity for edges of entire graph, individual layers
/// and `n` most dissimilar nodes for each individual layer
#[derive(Clone, Default)]
pub struct DetailedJaccardLD {
    pub n: usize,
}

impl<V: Ref + Display + FromStr> LayerDiffer<V> for DetailedJaccardLD {
    type LayerDiff = (JaccardState, Vec<(V, JaccardState)>);
    type ND = JaccardND;

    fn accumulate(
        &self,
        per_nb: Vec<(V, <Self::ND as NeighborhoodDiffer<V>>::NeighborhoodDiff)>,
    ) -> Self::LayerDiff {
        let mut per_nb = per_nb;
        per_nb.sort_by(|lhs, rhs| lhs.1.compare_as_fractions(&rhs.1));
        let mut acc = JaccardState::default();
        for (_, val) in per_nb.iter() {
            acc = acc + val.clone();
        }
        per_nb.truncate(self.n);

        (acc, per_nb)
    }
}

pub struct DetailedJaccard {
    pub n: usize,
}

impl<V: Ref + Display + FromStr> GraphDiffer<V> for DetailedJaccard {
    type GraphDiff = (JaccardState, Vec<(JaccardState, Vec<(V, JaccardState)>)>);
    type LD = DetailedJaccardLD;

    fn accumulate(
        &self,
        per_layer: Vec<<Self::LD as LayerDiffer<V>>::LayerDiff>,
    ) -> Self::GraphDiff {
        let mut acc = JaccardState::default();
        for (lacc, _) in per_layer.iter() {
            acc = acc + lacc.clone();
        }
        (acc, per_layer)
    }
}
