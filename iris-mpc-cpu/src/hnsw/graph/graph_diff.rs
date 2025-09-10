use std::collections::{HashMap, HashSet};
use std::fmt;

// Assuming these are defined elsewhere in your crate.
// You might need to adjust the `use` paths.
use crate::hnsw::{graph::layered_graph::Layer, GraphMem, VectorStore};

// --- 1. Neighborhood Diffing ---

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

// --- 2. Layer Diffing (Modified) ---

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
    ///
    /// ## Panics
    /// This function ASSUMES (and is guaranteed by `diff_graph`) that both layers
    /// have identical node sets. It will panic if a node key from `self` is not
    /// found in `other`.
    pub fn diff_layer(&self, other: &Layer<V>) -> LayerDiff<V> {
        let mut total_common_edges = 0;
        let mut total_edges_in_self = 0;
        let mut total_edges_in_other = 0;
        let mut percentage_sum = 0.0;
        let mut per_node_diffs = HashMap::new();

        // We can just iterate over self's keys, as `diff_graph` has guaranteed
        // the key sets of both layers are identical.
        let num_neighborhoods = self.links.len();

        for node_ref in self.links.keys() {
            // This should always return Some, based on the guarantee from diff_graph.
            if let Some(neighborhood_diff) = self.diff_neighborhood(other, node_ref) {
                total_common_edges += neighborhood_diff.common_edges;
                total_edges_in_self += neighborhood_diff.total_edges_in_self;
                total_edges_in_other += neighborhood_diff.total_edges_in_other;
                percentage_sum += neighborhood_diff.percentage_common;
                per_node_diffs.insert(node_ref.clone(), neighborhood_diff);
            } else {
                // This case should be impossible if the graph diff pre-check passed.
                unreachable!(
                    "Graph diff logic failure: Node {:?} in self layer keySet but not in other",
                    node_ref
                );
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

// --- 3. Graph Diffing (Modified) ---

/// Holds validation results and (if valid) the diff of two graphs.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphDiff<V: VectorStore> {
    // --- Validation Fields ---
    /// Number of layers in the `self` graph.
    pub self_layer_count: usize,
    /// Number of layers in the `other` graph.
    pub other_layer_count: usize,
    /// List of layer indices where the set of nodes (vertices) did not match.
    pub mismatched_node_set_layers: Vec<usize>,

    // --- Diff Fields (Only populated if validation passes) ---
    pub total_common_edges: usize,
    pub total_edges_in_self: usize,
    pub total_edges_in_other: usize,
    /// The overall percentage of common edges across all layers.
    pub overall_percentage_common: f64,
    pub per_layer_diffs: Vec<LayerDiff<V>>,
}

impl<V: VectorStore> GraphDiff<V> {
    /// Helper to create a new, empty diff, primarily for returning early on failure.
    fn new_aborted(self_layers: usize, other_layers: usize, mismatched_nodes: Vec<usize>) -> Self {
        GraphDiff {
            self_layer_count: self_layers,
            other_layer_count: other_layers,
            mismatched_node_set_layers: mismatched_nodes,
            total_common_edges: 0,
            total_edges_in_self: 0,
            total_edges_in_other: 0,
            overall_percentage_common: 0.0,
            per_layer_diffs: Vec::new(),
        }
    }

    /// Returns true if the diff was aborted due to validation failure.
    pub fn did_fail_validation(&self) -> bool {
        self.self_layer_count != self.other_layer_count
            || !self.mismatched_node_set_layers.is_empty()
    }
}

impl<V: VectorStore> fmt::Display for GraphDiff<V>
where
    V::VectorRef: fmt::Debug + Ord, // Added Ord for stable sorting of nodes
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // --- 1. Print Validation Failure ---
        if self.self_layer_count != self.other_layer_count {
            writeln!(f, "🛑 Graph Diff Aborted: Layer count mismatch.")?;
            writeln!(f, "   Self:  {} layers", self.self_layer_count)?;
            writeln!(f, "   Other: {} layers", self.other_layer_count)?;
            return Ok(());
        }

        if !self.mismatched_node_set_layers.is_empty() {
            writeln!(f, "🛑 Graph Diff Aborted: Node sets mismatch.")?;
            writeln!(
                f,
                "   The following layers do not contain the exact same set of nodes:"
            )?;
            for layer_index in &self.mismatched_node_set_layers {
                writeln!(f, "     - Layer {}", layer_index)?;
            }
            return Ok(());
        }

        // --- 2. Print Full Diff Report (Validation Passed) ---
        writeln!(f, "✅ Graph Diff Summary (Structural Match):")?;
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

            writeln!(f, "  Top 20 Nodes with < 100% Commonality:")?;

            let mut sorted_nodes: Vec<_> = layer_diff.per_node_diffs.iter().collect();
            // Sort by percentage_common ascending, then by node_ref for stable sort
            sorted_nodes.sort_by(|(ka, a), (kb, b)| {
                a.percentage_common
                    .partial_cmp(&b.percentage_common)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| ka.cmp(kb)) // Use node ref for tie-breaking
            });

            // Filter for problematic nodes *before* taking 20
            let problematic_nodes: Vec<_> = sorted_nodes
                .into_iter()
                .filter(|(_, diff)| diff.percentage_common < 100.0)
                .collect();

            if problematic_nodes.is_empty() {
                writeln!(f, "    All nodes in this layer have 100% commonality.")?;
            } else {
                for (node_ref, diff) in problematic_nodes.iter().take(20) {
                    writeln!(
                        f,
                        "    Node {:?}: {:.2}% common ({}/{}) [Self: {}, Other: {}]",
                        node_ref,
                        diff.percentage_common,
                        diff.common_edges,
                        diff.total_edges_in_self + diff.total_edges_in_other - diff.common_edges, // Total unique edges
                        diff.total_edges_in_self,
                        diff.total_edges_in_other
                    )?;
                }
            }
        }

        Ok(())
    }
}

impl<V: VectorStore> GraphMem<V> {
    /// Diffs this graph against another, comparing layer by layer.
    /// First validates that layer counts and all node sets per layer are identical.
    /// If validation fails, returns a GraphDiff struct containing only the failure info.
    pub fn diff_graph(&self, other: &GraphMem<V>) -> GraphDiff<V> {
        let self_layer_count = self.layers.len();
        let other_layer_count = other.layers.len();

        // --- Validation Check 1: Layer Count ---
        if self_layer_count != other_layer_count {
            return GraphDiff::new_aborted(self_layer_count, other_layer_count, vec![]);
        }

        // --- Validation Check 2: Node Sets Per Layer ---
        let mut mismatched_node_set_layers = Vec::new();
        for i in 0..self_layer_count {
            // These indexing ops are safe due to the check above.
            let self_nodes: HashSet<_> = self.layers[i].links.keys().collect();
            let other_nodes: HashSet<_> = other.layers[i].links.keys().collect();

            if self_nodes != other_nodes {
                mismatched_node_set_layers.push(i);
            }
        }

        if !mismatched_node_set_layers.is_empty() {
            return GraphDiff::new_aborted(
                self_layer_count,
                other_layer_count,
                mismatched_node_set_layers,
            );
        }

        // --- Validation Passed: Proceed with Full Diff ---
        let mut total_common_edges = 0;
        let mut total_edges_in_self = 0;
        let mut total_edges_in_other = 0;
        let mut per_layer_diffs = Vec::new();

        for i in 0..self_layer_count {
            // We know layers exist and node sets match.
            let self_layer = &self.layers[i];
            let other_layer = &other.layers[i];

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
            (total_common_edges as f64 / total_unique_edges as f64) * 100.0
        };

        GraphDiff {
            self_layer_count,
            other_layer_count,
            mismatched_node_set_layers: Vec::new(), // Success, so empty
            total_common_edges,
            total_edges_in_self,
            total_edges_in_other,
            overall_percentage_common,
            per_layer_diffs,
        }
    }
}
