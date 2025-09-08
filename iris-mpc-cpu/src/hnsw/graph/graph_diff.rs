// --- 1. Neighborhood Diffing ---

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::hnsw::{graph::layered_graph::Layer, GraphMem, VectorStore};

/// Holds the result of diffing two neighborhoods for a single node.
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborhoodDiff {
    pub common_edges: usize,
    pub total_edges_in_self: usize,
    pub total_edges_in_other: usize,
    /// Percentage of common edges relative to the total number of unique edges.
    pub percentage_common: f64,
}

impl<V: VectorStore> Layer<V> {
    /// Diffs the neighborhood of a specific node between this layer and another.
    pub fn diff_neighborhood(
        &self,
        other: &Layer<V>,
        node: &V::VectorRef,
    ) -> Option<NeighborhoodDiff> {
        let self_neighbors = self.links.get(node);
        let other_neighbors = other.links.get(node);

        match (self_neighbors, other_neighbors) {
            (Some(self_nb), Some(other_nb)) => {
                let self_set: HashSet<_> = self_nb.iter().collect();
                let other_set: HashSet<_> = other_nb.iter().collect();

                let common_edges = self_set.intersection(&other_set).count();
                let total_unique_edges = self_set.union(&other_set).count();

                let percentage_common = if total_unique_edges == 0 {
                    100.0 // Both are empty, so they are perfectly matching
                } else {
                    (common_edges as f64 / total_unique_edges as f64) * 100.0
                };

                Some(NeighborhoodDiff {
                    common_edges,
                    total_edges_in_self: self_set.len(),
                    total_edges_in_other: other_set.len(),
                    percentage_common,
                })
            }
            // Return None if the node doesn't exist in both layers for comparison.
            _ => None,
        }
    }
}

// --- 2. Layer Diffing ---

/// Holds the aggregated result of diffing two layers.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDiff<V: VectorStore> {
    pub total_common_edges: usize,
    pub total_edges_in_self: usize,
    pub total_edges_in_other: usize,
    /// Average percentage of common edges across all neighborhoods in the layer.
    pub avg_percentage_common: f64,
    /// Holds individual diffs for each node in the layer for detailed debugging.
    pub per_node_diffs: HashMap<V::VectorRef, NeighborhoodDiff>,
}

impl<V: VectorStore> Layer<V> {
    /// Diffs this entire layer against another layer.
    pub fn diff_layer(&self, other: &Layer<V>) -> LayerDiff<V> {
        let mut total_common_edges = 0;
        let mut total_edges_in_self = 0;
        let mut total_edges_in_other = 0;
        let mut percentage_sum = 0.0;
        let mut per_node_diffs = HashMap::new();

        // Use a set of all nodes from both layers to ensure we check every neighborhood.
        let all_nodes: HashSet<_> = self.links.keys().chain(other.links.keys()).collect();
        let num_neighborhoods = all_nodes.len();

        for node_ref in all_nodes {
            if let Some(neighborhood_diff) = self.diff_neighborhood(other, node_ref) {
                total_common_edges += neighborhood_diff.common_edges;
                total_edges_in_self += neighborhood_diff.total_edges_in_self;
                total_edges_in_other += neighborhood_diff.total_edges_in_other;
                percentage_sum += neighborhood_diff.percentage_common;
                per_node_diffs.insert(node_ref.clone(), neighborhood_diff);
            } else {
                // If a node exists in one but not the other, we count its edges
                // and store a 0%-common diff for debugging.
                let self_nb_len = self.links.get(node_ref).map_or(0, |nb| nb.len());
                let other_nb_len = other.links.get(node_ref).map_or(0, |nb| nb.len());

                total_edges_in_self += self_nb_len;
                total_edges_in_other += other_nb_len;

                let node_only_diff = NeighborhoodDiff {
                    common_edges: 0,
                    total_edges_in_self: self_nb_len,
                    total_edges_in_other: other_nb_len,
                    percentage_common: 0.0, // Commonality is 0%
                };
                per_node_diffs.insert(node_ref.clone(), node_only_diff);
            }
        }

        let avg_percentage_common = if num_neighborhoods == 0 {
            100.0
        } else {
            percentage_sum / num_neighborhoods as f64
        };

        LayerDiff {
            total_common_edges,
            total_edges_in_self,
            total_edges_in_other,
            avg_percentage_common,
            per_node_diffs,
        }
    }
}

// --- 3. Graph Diffing ---

/// Holds the final aggregated result of diffing two graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDiff<V: VectorStore> {
    pub total_common_edges: usize,
    pub total_edges_in_self: usize,
    pub total_edges_in_other: usize,
    /// The overall percentage of common edges across all layers.
    pub overall_percentage_common: f64,
    pub per_layer_diffs: Vec<LayerDiff<V>>,
}

impl<V: VectorStore> fmt::Display for GraphDiff<V>
where
    V::VectorRef: fmt::Debug, // Required to print the node reference
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph Diff Summary:")?;
        writeln!(
            f,
            "  Overall Commonality: {:.2}%",
            self.overall_percentage_common
        )?;
        writeln!(f, "  Total Edges (Self):  {}", self.total_edges_in_self)?;
        writeln!(f, "  Total Edges (Other): {}", self.total_edges_in_other)?;
        writeln!(f, "  Total Common Edges:  {}", self.total_common_edges)?;

        for (i, layer_diff) in self.per_layer_diffs.iter().enumerate() {
            writeln!(f, "\n--- Layer {} ---", i)?;
            writeln!(
                f,
                "  Layer Avg. Commonality: {:.2}%",
                layer_diff.avg_percentage_common
            )?;
            writeln!(
                f,
                "  Edges (Self): {}, Edges (Other): {}, Common: {}",
                layer_diff.total_edges_in_self,
                layer_diff.total_edges_in_other,
                layer_diff.total_common_edges
            )?;

            if layer_diff.per_node_diffs.is_empty() {
                writeln!(f, "  No nodes to compare in this layer.")?;
                continue;
            }

            writeln!(f, "  Top 100 Nodes with Lowest Neighborhood Commonality:")?;

            let mut sorted_nodes: Vec<_> = layer_diff.per_node_diffs.iter().collect();
            // Sort by percentage_common ascending. Using partial_cmp as f64 is not Ord.
            sorted_nodes.sort_by(|(_, a), (_, b)| {
                a.percentage_common
                    .partial_cmp(&b.percentage_common)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for (node_ref, diff) in sorted_nodes.iter().take(100) {
                writeln!(
                    f,
                    "    Node {:?}: {:.2}% common ({}/{}) [Self: {}, Other: {}]",
                    node_ref,
                    diff.percentage_common,
                    diff.common_edges,
                    diff.total_edges_in_self + diff.total_edges_in_other - diff.common_edges,
                    diff.total_edges_in_self,
                    diff.total_edges_in_other
                )?;
            }
        }

        Ok(())
    }
}

impl<V: VectorStore> GraphMem<V> {
    /// Diffs this graph against another, comparing layer by layer.
    pub fn diff_graph(&self, other: &GraphMem<V>) -> GraphDiff<V> {
        let mut total_common_edges = 0;
        let mut total_edges_in_self = 0;
        let mut total_edges_in_other = 0;
        let mut per_layer_diffs = Vec::new();

        let num_layers = self.layers.len().max(other.layers.len());

        for i in 0..num_layers {
            let empty_layer = Layer::new();
            let self_layer = self.layers.get(i).unwrap_or(&empty_layer);
            let other_layer = other.layers.get(i).unwrap_or(&empty_layer);

            let layer_diff = self_layer.diff_layer(other_layer);

            total_common_edges += layer_diff.total_common_edges;
            total_edges_in_self += layer_diff.total_edges_in_self;
            total_edges_in_other += layer_diff.total_edges_in_other;

            per_layer_diffs.push(layer_diff);
        }

        let total_unique_edges = total_edges_in_self + total_edges_in_other - total_common_edges;
        let overall_percentage_common = if total_unique_edges == 0 {
            100.0
        } else {
            // Note: This calculates common edges as a percentage of *unique* edges,
            // which is a Jaccard-like similarity.
            (total_common_edges as f64 / total_unique_edges as f64) * 100.0
        };

        GraphDiff {
            total_common_edges,
            total_edges_in_self,
            total_edges_in_other,
            overall_percentage_common,
            per_layer_diffs,
        }
    }
}
