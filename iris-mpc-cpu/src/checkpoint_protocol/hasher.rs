//! `Blake3GraphHasher` — streaming BLAKE3 over the canonical bytes of a
//! `Graph` (i.e. `BothEyes<GraphMem<VectorId>>`).
//!
//! # Canonical bytes contract
//!
//! Three parties materialize the same WAL replay independently and must
//! produce byte-identical serializations for hash consensus to work. The
//! canonical bytes are defined as `bincode::serialize(&graph)` where `graph:
//! &BothEyes<GraphMem<VectorId>>`, with these determinism guarantees:
//!
//! 1. **Eye order**: `[LEFT, RIGHT]`, by the existing `BothEyes<T> = [T; 2]`
//!    convention.
//! 2. **`GraphMem.layers`** is a `Vec<Layer<V>>` indexed by layer number; Vec
//!    iteration order matches layer index, deterministic by construction.
//! 3. **`GraphMem.entry_points`** is a `Vec<EntryPoint<V>>` in the order the
//!    parties applied entry-point operations. Because all parties replay the
//!    same WAL in the same order, this Vec ends up identical across parties.
//! 4. **`Layer.links: HashMap<V, Vec<V>>`** is serialized via the custom
//!    [`SortedLinks`] wrapper at `layered_graph.rs:380`, which sorts entries
//!    by key before emitting. HashMap iteration order is therefore *not* a
//!    determinism risk at serialize time.
//! 5. **Per-key neighbor `Vec<V>`** is emitted in insertion order. The
//!    invariant is that the planner sorts neighbors before calling
//!    `set_links`, and `Layer::add_neighbor` maintains sortedness via
//!    `binary_search + insert`. Any caller bypassing the planner can break
//!    this invariant and corrupt hash consensus.
//! 6. **`Layer.set_hash: SetHash`** is order-agnostic by design (wrapping
//!    add + SipHash13). Trivially deterministic given the final
//!    `(key, value)` pairs.
//!
//! Of these, only (5) depends on a load-bearing invariant in the planner
//! rather than the serializer itself. If neighbor lists ever get out of
//! order, hash consensus fails immediately (a fatal cycle error) — loud
//! rather than silent. A debug-only assertion that neighbor lists are
//! sorted at serialize time is a candidate follow-up.
//!
//! # Streaming
//!
//! `blake3::Hasher` implements `std::io::Write`, so `bincode::serialize_into`
//! feeds bytes straight into the hasher with no intermediate buffer. The
//! hash bytes the existing buffered `blake3::hash(bincode::serialize(...))`
//! call produces are bit-identical to what this hasher returns.

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
    use iris_mpc_common::vector_id::VectorId;
    use std::collections::HashMap;

    fn vid(i: u32) -> VectorId {
        VectorId::from_0_index(i)
    }

    type LayerInput = Vec<(usize, Vec<(VectorId, Vec<VectorId>)>)>;

    /// Hand-build a graph with controlled state. Skipping `set_links` etc.
    /// lets the tests check serializer-level determinism rather than the
    /// planner's sortedness invariant.
    fn graph_from(eyes: [LayerInput; 2]) -> Graph {
        let build = |layers_in: LayerInput| -> GraphMem<VectorId> {
            let mut g = GraphMem::<VectorId>::new();
            let max_lc = layers_in.iter().map(|(lc, _)| *lc).max().unwrap_or(0);
            let mut layers: Vec<Layer<VectorId>> =
                (0..=max_lc).map(|_| Layer::<VectorId>::new()).collect();
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

    /// HashMap iteration order is unspecified across runs; the custom
    /// `SortedLinks` Serialize impl on `Layer` is what makes the hash
    /// deterministic. This test simulates two parties' HashMaps populated in
    /// different orders and asserts the canonical bytes match.
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

    /// The hash is BLAKE3 of the bincode bytes — wire-compatible with the
    /// existing buffered `upload_graph_checkpoint` path's hex hash. If this
    /// equality ever breaks, every checkpoint written by the new pipeline
    /// will be unverifiable by readers using the old format. Lock it in.
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
        let g1: Graph = [GraphMem::<VectorId>::new(), GraphMem::<VectorId>::new()];
        let g2: Graph = [GraphMem::<VectorId>::new(), GraphMem::<VectorId>::new()];
        let h = Blake3GraphHasher::new();
        assert_eq!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }

    #[test]
    fn left_right_swap_changes_hash() {
        let left = {
            let mut l = Layer::<VectorId>::new();
            l.set_links(vid(1), vec![vid(2)]);
            let mut g = GraphMem::<VectorId>::new();
            g.layers = vec![l];
            g
        };
        let right = {
            let mut l = Layer::<VectorId>::new();
            l.set_links(vid(5), vec![vid(6)]);
            let mut g = GraphMem::<VectorId>::new();
            g.layers = vec![l];
            g
        };
        let g1: Graph = [left.clone(), right.clone()];
        let g2: Graph = [right, left];
        let h = Blake3GraphHasher::new();
        assert_ne!(h.hash_canonical(&g1), h.hash_canonical(&g2));
    }

    /// Suppress the unused-import warning in this test module. The HashMap
    /// import is for the inline `graph_from` helper above.
    #[allow(dead_code)]
    fn _force_use() {
        let _: HashMap<u8, u8> = HashMap::new();
    }
}
