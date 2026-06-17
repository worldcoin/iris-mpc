//! Streaming BLAKE3 over the canonical bytes of a `Graph`.
//!
//! Cross-party determinism depends on two non-obvious invariants:
//!
//! - **`Layer.links: HashMap<V, Vec<V>>`** is serialized via the custom
//!   `SortedLinks` wrapper in `layered_graph.rs`, which sorts entries by
//!   key before emitting — so HashMap iteration order is not a hash risk.
//! - **Per-key neighbor `Vec<V>`** is emitted in insertion order; the
//!   planner is responsible for sorting neighbors before `set_links`, and
//!   `Layer::add_neighbor` maintains sortedness. Bypassing the planner
//!   breaks hash consensus.

use crate::checkpoint_protocol::{Blake3Hash, Graph, GraphHasher};

#[derive(Default, Clone, Copy)]
pub struct Blake3GraphHasher;

impl Blake3GraphHasher {
    pub fn new() -> Self {
        Self
    }
}

impl GraphHasher for Blake3GraphHasher {
    fn hash_canonical(&self, graph: &Graph) -> Blake3Hash {
        let mut hasher = blake3::Hasher::new();
        bincode::serialize_into(&mut hasher, graph)
            .expect("bincode::serialize_into cannot fail on blake3::Hasher");
        *hasher.finalize().as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::graph::layered_graph::{EntryPoint, GraphMem, Layer};
    use iris_mpc_common::VectorId;

    fn vid(i: u32) -> VectorId {
        VectorId::from_0_index(i)
    }

    type LayerInput = Vec<(usize, Vec<(VectorId, Vec<VectorId>)>)>;

    /// Bypasses the planner so tests can pin serializer-level determinism
    /// independently of the planner's sortedness invariant.
    fn graph_from(eyes: [LayerInput; 2]) -> Graph {
        let build = |layers_in: LayerInput| -> GraphMem {
            let mut g = GraphMem::new();
            let max_lc = layers_in.iter().map(|(lc, _)| *lc).max().unwrap_or(0);
            let mut layers: Vec<Layer> = (0..=max_lc).map(|_| Layer::new()).collect();
            for (lc, pairs) in layers_in {
                for (k, v) in pairs {
                    layers[lc].set_links(k, v);
                }
            }
            g.layers = layers;
            g.entry_points = vec![EntryPoint {
                point: vid(0),
                layer: 0,
            }];
            g
        };
        [build(eyes[0].clone()), build(eyes[1].clone())]
    }

    #[test]
    fn hash_is_deterministic_on_identical_graphs() {
        let g1 = graph_from([
            vec![(
                0,
                vec![
                    (vid(1), vec![vid(2), vid(3)]),
                    (vid(2), vec![vid(1), vid(3)]),
                ],
            )],
            vec![(0, vec![(vid(5), vec![vid(6)])])],
        ]);
        let g2 = g1.clone();

        let h = Blake3GraphHasher::new();
        assert_eq!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }

    #[test]
    fn hash_changes_when_a_neighbor_changes() {
        let g1 = graph_from([
            vec![(0, vec![(vid(1), vec![vid(2)])])],
            vec![(0, vec![(vid(5), vec![vid(6)])])],
        ]);
        let g2 = graph_from([
            vec![(0, vec![(vid(1), vec![vid(3)])])], // vid(3) instead of vid(2)
            vec![(0, vec![(vid(5), vec![vid(6)])])],
        ]);

        let h = Blake3GraphHasher::new();
        assert_ne!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }

    /// Load-bearing: pins the `SortedLinks` Serialize impl that makes
    /// HashMap iteration order irrelevant to the hash.
    #[test]
    fn hash_is_independent_of_hashmap_insertion_order() {
        let entries: Vec<(VectorId, Vec<VectorId>)> = (0..32)
            .map(|i| (vid(i), vec![vid(i + 100), vid(i + 200)]))
            .collect();

        let g_forward = graph_from([vec![(0, entries.clone())], vec![(0, entries.clone())]]);
        let g_reversed = graph_from([
            vec![(0, entries.iter().rev().cloned().collect())],
            vec![(0, entries.iter().rev().cloned().collect())],
        ]);

        let h = Blake3GraphHasher::new();
        assert_eq!(h.hash_canonical(&g_forward), h.hash_canonical(&g_reversed));
    }

    /// Wire-compat gate: the streaming hash must equal the buffered
    /// `blake3::hash(bincode::serialize(&graph))` used by the existing
    /// `upload_graph_checkpoint` path.
    #[test]
    fn streaming_hash_matches_buffered_blake3_of_bincode_bytes() {
        let g = graph_from([
            vec![
                (
                    0,
                    vec![(vid(1), vec![vid(2), vid(3)]), (vid(2), vec![vid(1)])],
                ),
                (1, vec![(vid(1), vec![])]),
            ],
            vec![(0, vec![(vid(5), vec![vid(6)])])],
        ]);

        let streamed = Blake3GraphHasher::new().hash_canonical(&g);
        let buffered = {
            let bytes = bincode::serialize(&g).expect("bincode");
            *blake3::hash(&bytes).as_bytes()
        };
        assert_eq!(streamed, buffered);
    }

    #[test]
    fn empty_graphs_hash_consistently() {
        let g1: Graph = [GraphMem::new(), GraphMem::new()];
        let g2: Graph = [GraphMem::new(), GraphMem::new()];
        let h = Blake3GraphHasher::new();
        assert_eq!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }

    #[test]
    fn left_right_swap_changes_hash() {
        let left = {
            let mut l = Layer::new();
            l.set_links(vid(1), vec![vid(2)]);
            let mut g = GraphMem::new();
            g.layers = vec![l];
            g
        };
        let right = {
            let mut l = Layer::new();
            l.set_links(vid(5), vec![vid(6)]);
            let mut g = GraphMem::new();
            g.layers = vec![l];
            g
        };
        let g1: Graph = [left.clone(), right.clone()];
        let g2: Graph = [right, left];
        let h = Blake3GraphHasher::new();
        assert_ne!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }
}
