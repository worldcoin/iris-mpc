use super::Differ;
use crate::hnsw::vector_store::Ref;
use std::{collections::HashSet, fmt::Display, ops::Add, str::FromStr};

#[derive(Debug, Default, Clone, Copy)]
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

    pub fn is_one(&self) -> bool {
        self.union == self.intersection
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

/// A differ that computes detailed Jaccard similarity, including the `n` most dissimilar nodes per layer.
pub struct DetailedJaccardDiffer<V: Ref> {
    n: usize,
    graph_state: JaccardState,
    current_layer_details: Vec<(V, JaccardState)>,
    per_layer_results: Vec<(JaccardState, Vec<(V, JaccardState)>)>,
}

impl<V: Ref> DetailedJaccardDiffer<V> {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            graph_state: JaccardState::default(),
            current_layer_details: Vec::new(),
            per_layer_results: Vec::new(),
        }
    }
}

pub struct DetailedJaccardReport<V>(
    #[allow(clippy::type_complexity)]
    pub  (JaccardState, Vec<(JaccardState, Vec<(V, JaccardState)>)>),
);

impl<V: Ref + Display> Display for DetailedJaccardReport<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (graph_state, per_layer_results) = &self.0;
        writeln!(f, "GRAPH aggregate: {}", graph_state)?;
        for (layer_idx, layer_result) in per_layer_results.iter().enumerate() {
            writeln!(f, "  LAYER {} aggregate: {}", layer_idx, layer_result.0)?;
            writeln!(f, "  Top {} most dissimilar nodes:", layer_result.1.len())?;
            for (node, node_state) in &layer_result.1 {
                writeln!(f, "    Node {}: {}", node, node_state)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<V: Ref + Display + FromStr> Differ<V> for DetailedJaccardDiffer<V> {
    type Output = DetailedJaccardReport<V>;

    fn start_layer(&mut self, _layer_index: usize) {
        self.current_layer_details.clear();
    }

    fn diff_neighborhood(&mut self, _layer_index: usize, node: &V, lhs: &[V], rhs: &[V]) {
        let lhs_set: HashSet<_> = lhs.iter().collect();
        let rhs_set: HashSet<_> = rhs.iter().collect();
        let intersection = lhs_set.intersection(&rhs_set).count();
        let union = lhs_set.union(&rhs_set).count();
        let node_state = JaccardState::new(intersection, union);
        self.current_layer_details.push((node.clone(), node_state));
    }

    fn end_layer(&mut self, _layer_index: usize) {
        let layer_total_state = self
            .current_layer_details
            .iter()
            .fold(JaccardState::default(), |acc, (_, state)| acc + *state);
        self.graph_state = self.graph_state + layer_total_state;

        self.current_layer_details
            .sort_by(|a, b| a.1.compare_as_fractions(&b.1));
        self.current_layer_details.truncate(self.n);

        self.per_layer_results.push((
            layer_total_state,
            std::mem::take(&mut self.current_layer_details),
        ));
    }

    fn finish(self) -> Self::Output {
        DetailedJaccardReport((self.graph_state, self.per_layer_results))
    }
}
