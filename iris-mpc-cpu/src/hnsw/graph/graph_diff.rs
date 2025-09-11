use crate::hnsw::{vector_store::Ref, GraphMem};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::Add;
use std::str::FromStr;
// --- 1. Information Collector Structures ---

/// Raw neighborhood information for a single node from two graphs.
pub struct NeighborhoodInfo<'a, V: Ref> {
    pub self_neighbors: HashSet<&'a V>,
    pub other_neighbors: HashSet<&'a V>,
}

/// A collection of all neighborhood information for a single layer.
pub struct LayerInfo<'a, V: Ref> {
    pub nodes: HashMap<&'a V, NeighborhoodInfo<'a, V>>,
}

/// A collection of all layer information for an entire graph.
pub struct GraphInfo<'a, V: Ref> {
    pub layers: Vec<LayerInfo<'a, V>>,
}

// --- 2. Generic Diffing Trait ---

/// A trait that defines a generic graph diffing algorithm.
/// It separates the logic of how to compare nodes/layers from the data collection.
pub trait EdgeDiffer<V: Ref + Display + FromStr> {
    /// The output of diffing a single neighborhood.
    type NeighborhoodDiff: Debug + Clone;
    /// The output of diffing an entire layer.
    type LayerDiff: Debug + Clone;
    /// The final output of diffing the graph, which must be displayable.
    type GraphDiff: Debug + Clone;

    /// Diffs a single neighborhood based on the provided info.
    fn diff_neighborhood(&self, info: &NeighborhoodInfo<V>) -> Self::NeighborhoodDiff;

    /// Diffs a layer by processing all its neighborhood info.
    fn diff_layer(&self, info: &LayerInfo<V>) -> Self::LayerDiff;

    /// Diffs a graph by processing all its layer info.
    fn diff_graph(&self, info: &GraphInfo<V>) -> Self::GraphDiff;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
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
    pub fn similarity(&self) -> f64 {
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

impl Sum for JaccardState {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(JaccardState::default(), |acc, x| acc + x)
    }
}

struct JaccardSimilarity;

impl<V: Ref + FromStr + Display> EdgeDiffer<V> for JaccardSimilarity {
    type NeighborhoodDiff = JaccardState;
    type LayerDiff = JaccardState;
    type GraphDiff = JaccardState;

    fn diff_neighborhood(&self, info: &NeighborhoodInfo<V>) -> Self::NeighborhoodDiff {
        JaccardState::new(
            info.other_neighbors
                .intersection(&info.self_neighbors)
                .count(),
            info.self_neighbors.union(&info.other_neighbors).count(),
        )
    }

    fn diff_layer(&self, info: &LayerInfo<V>) -> Self::LayerDiff {
        info.nodes
            .iter()
            .map(|(_, ret)| self.diff_neighborhood(ret))
            .sum()
    }

    fn diff_graph(&self, info: &GraphInfo<V>) -> Self::GraphDiff {
        info.layers.iter().map(|ret| self.diff_layer(ret)).sum()
    }
}

// --- 3. Public API & Validation Structures ---

/// Contains detailed information about why a diff failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDiffFailure {
    pub reason: GraphDiffFailureReason,
}

/// Specific reasons for a graph diff validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphDiffFailureReason {
    LayerCountMismatch {
        self_layers: usize,
        other_layers: usize,
    },
    NodeSetMismatch {
        mismatched_layers: Vec<usize>,
    },
}

/// The public entry point for diffing two graphs.
///
/// It first validates the graphs, then collects the raw link information,
/// and finally uses the provided `differ` to compute and return the result.
pub fn diff_graph<V, D>(
    self_graph: &GraphMem<V>,
    other_graph: &GraphMem<V>,
    differ: D,
) -> Result<D::GraphDiff, GraphDiffFailure>
where
    V: Ref + Display + FromStr,
    D: EdgeDiffer<V>,
{
    // --- Validation Step ---
    if self_graph.layers.len() != other_graph.layers.len() {
        return Err(GraphDiffFailure {
            reason: GraphDiffFailureReason::LayerCountMismatch {
                self_layers: self_graph.layers.len(),
                other_layers: other_graph.layers.len(),
            },
        });
    }

    let mismatched_layers: Vec<usize> = (0..self_graph.layers.len())
        .filter(|&i| {
            let self_nodes: HashSet<_> = self_graph.layers[i].links.keys().collect();
            let other_nodes: HashSet<_> = other_graph.layers[i].links.keys().collect();
            self_nodes != other_nodes
        })
        .collect();

    if !mismatched_layers.is_empty() {
        return Err(GraphDiffFailure {
            reason: GraphDiffFailureReason::NodeSetMismatch { mismatched_layers },
        });
    }

    // --- Information Collection Step ---
    let graph_info = GraphInfo {
        layers: self_graph
            .layers
            .iter()
            .zip(other_graph.layers.iter())
            .map(|(self_layer, other_layer)| {
                let nodes = self_layer
                    .links
                    .keys()
                    .map(|node_ref| {
                        let self_neighbors = self_layer
                            .links
                            .get(node_ref)
                            .map_or(HashSet::new(), |v| v.iter().collect());
                        let other_neighbors = other_layer
                            .links
                            .get(node_ref)
                            .map_or(HashSet::new(), |v| v.iter().collect());

                        let info = NeighborhoodInfo {
                            self_neighbors,
                            other_neighbors,
                        };
                        (node_ref, info)
                    })
                    .collect();
                LayerInfo { nodes }
            })
            .collect(),
    };

    // --- Diffing Step ---
    Ok(differ.diff_graph(&graph_info))
}
