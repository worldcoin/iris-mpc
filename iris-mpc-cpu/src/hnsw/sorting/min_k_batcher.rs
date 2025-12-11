//! Implementation of Batcher Odd-Even Swap Network adapted for selecting
//! the smallest K values (in arbitrary order).
//!
//! Based on the following paper:
//! - (<https://eprint.iacr.org/2023/852.pdf>) [1]

use cached::proc_macro::cached;
use eyre::{bail, OptionExt, Result};
use std::{
    cmp,
    collections::{BTreeMap, HashSet},
};

use crate::hnsw::sorting::swap_network::SwapNetwork;

struct MinKBatcherBuilder {
    comparators_by_layer: BTreeMap<usize, HashSet<(usize, usize)>>,
}

impl MinKBatcherBuilder {
    fn new() -> Self {
        Self {
            comparators_by_layer: BTreeMap::new(),
        }
    }

    fn add_comparator(&mut self, layer: usize, idx1: usize, idx2: usize) {
        self.comparators_by_layer
            .entry(layer)
            .or_default()
            .insert((idx1, idx2));
    }

    /// Converts the internal state into the standard SwapNetwork format.
    fn to_swap_network(&self) -> SwapNetwork {
        let mut layers = Vec::new();

        // If map is empty, return empty network
        if self.comparators_by_layer.is_empty() {
            return SwapNetwork { layers };
        }

        let max_layer = *self.comparators_by_layer.keys().max().unwrap_or(&0);

        // Ensure we have vector entries for every layer up to max
        for i in 0..=max_layer {
            if let Some(comps) = self.comparators_by_layer.get(&i) {
                let mut layer_vec: Vec<(usize, usize)> = comps.iter().cloned().collect();
                layer_vec.sort();
                layers.push(layer_vec);
            } else {
                // Empty intermediate layers are possible in recursive logic,
                // though usually collapsed in final output.
                layers.push(vec![]);
            }
        }

        // Filter out trailing empty layers if any
        while let Some(last) = layers.last() {
            if last.is_empty() {
                layers.pop();
            } else {
                break;
            }
        }

        SwapNetwork { layers }
    }
}

/// Implements the ChunkSize function from Algorithm 3.
fn chunk_size(d: usize, k: usize) -> usize {
    if k == 0 {
        return d.div_ceil(2);
    }

    let mu = if k > 1 { 1 << ((k - 1).ilog2() + 1) } else { 1 };

    if d <= mu {
        d.div_ceil(2)
    } else {
        mu * d.div_ceil(2 * mu)
    }
}

/// - See Algorithm 2 in [1]
/// - Append layers which merge sequences `x` and `y`, assuming that these correspond to sorted values. Note that `x` and `y` are sequences of indices.
/// - `start_layer` points to the layer in the global network on top of which this method should build.
/// - Returns (indices which form the result, total depth of the built network)
fn build_recursive_merge(
    builder: &mut MinKBatcherBuilder,
    x: Vec<usize>,
    y: Vec<usize>,
    k: usize,
    start_layer: usize,
) -> (Vec<usize>, usize) {
    let d1 = x.len();
    let d2 = y.len();

    if d1 * d2 == 0 {
        let mut combined = x;
        combined.extend(y);
        combined.truncate(k);
        return (combined, 0);
    }

    if d1 * d2 == 1 {
        builder.add_comparator(start_layer, x[0], y[0]);
        let mut combined = vec![x[0], y[0]];
        combined.truncate(k);
        return (combined, 1);
    }

    // Split x and y into even/odd indices
    let even_x: Vec<usize> = x.iter().step_by(2).cloned().collect();
    let odd_x: Vec<usize> = x.iter().skip(1).step_by(2).cloned().collect();

    let even_y: Vec<usize> = y.iter().step_by(2).cloned().collect();
    let odd_y: Vec<usize> = y.iter().skip(1).step_by(2).cloned().collect();

    let (v_indices, v_depth) =
        build_recursive_merge(builder, even_x, even_y, (k / 2) + 1, start_layer);
    let (w_indices, w_depth) = build_recursive_merge(builder, odd_x, odd_y, k / 2, start_layer);

    let recursive_depth = cmp::max(v_depth, w_depth);

    // Interleave v and w to form z
    let mut z_indices = Vec::with_capacity(v_indices.len() + w_indices.len());
    let mut v_ptr = 0;
    let mut w_ptr = 0;
    while v_ptr < v_indices.len() || w_ptr < w_indices.len() {
        if v_ptr < v_indices.len() {
            z_indices.push(v_indices[v_ptr]);
            v_ptr += 1;
        }
        if w_ptr < w_indices.len() {
            z_indices.push(w_indices[w_ptr]);
            w_ptr += 1;
        }
    }

    let final_comp_layer = start_layer + recursive_depth;

    let mut i = 1;
    while i < z_indices.len() {
        if i + 1 < z_indices.len() {
            let idx1 = z_indices[i];
            let idx2 = z_indices[i + 1];
            builder.add_comparator(final_comp_layer, idx1, idx2);
        }
        i += 2;
    }

    let total_depth = recursive_depth + 1;
    z_indices.truncate(k);
    (z_indices, total_depth)
}

///  Alekseev's merge for the final top-level step.
///  Takes two sorted sequences `v_perm` and `w_perm` of length at most `k` and returns
///  the smallest `min(k, v_perm.len() + w_perm.len()` elements of `v_perm + w_perm` in no particular order.
///  For correctness:
///  - First analyze the case where `v_perm.len() == w_perm.len() == k`.
///  - In this case Alekseev just does `min_swap(v_perm[i], w_perm[k - i - 1])` for `0 <= i < k`.
///  - Proof by contradiction, assuming there exists some element in post-swaps `w_perm` which is smaller than some element in post-swaps `v_perm`
///  - Derive contradiction from inequalities given by sortedness of inputs + inequalities given by execution of swaps.
///  - To handle arbitrary `<= k` sizes, imagine appending `+inf` to `v_perm` up to length `k`, prepending `-inf` to `w_perm` up to length `k` and eliminating comparators which involve infinities.
fn alekseev_merge(
    builder: &mut MinKBatcherBuilder,
    v_perm: &[usize],
    w_perm: &[usize],
    k: usize,
    start_layer: usize,
) -> (Vec<usize>, usize) {
    let n = v_perm.len();
    let m = w_perm.len();

    assert!(n <= k && m <= k);

    let mut res_perm = Vec::new();

    for i in 0..k {
        if i < n && (k - i - 1) < m {
            let idx1 = v_perm[i];
            let idx2 = w_perm[k - i - 1];
            res_perm.push(idx1);
            builder.add_comparator(start_layer, idx1, idx2);
        } else if i < n && (k - i - 1) >= m {
            res_perm.push(v_perm[i]);
        } else if i >= n && (k - i - 1) < m {
            res_perm.push(w_perm[k - i - 1]);
        }
    }

    (res_perm, 1)
}

/// - Algorithm 3 in [1]
/// - Builds a network that selects the `k` smallest values corresponding to `x_indices`.
/// - `start_layer` points to the layer in the global network on top of which this method should build.
/// - `is_top_level` is tracked so that we can apply Alekseev's merge as an optimization.
/// - Returns (indices which form the result, total depth of the buitl network)
fn build_recursive_sort(
    builder: &mut MinKBatcherBuilder,
    x_indices: Vec<usize>,
    k: usize,
    start_layer: usize,
    is_top_level: bool,
) -> (Vec<usize>, usize) {
    if k == 0 {
        return (Vec::new(), 0);
    }

    let d = x_indices.len();
    if d <= 1 {
        return (x_indices, 0);
    }

    let chunk_idx = chunk_size(d, k);
    let (chunk1, chunk2) = x_indices.split_at(chunk_idx);

    let (v_perm, v_depth) = build_recursive_sort(builder, chunk1.to_vec(), k, start_layer, false);
    let (w_perm, w_depth) = build_recursive_sort(builder, chunk2.to_vec(), k, start_layer, false);

    let recursive_depth = cmp::max(v_depth, w_depth);
    let merge_start_layer = start_layer + recursive_depth;

    // Note that Alekseev's merge is only done at the top level
    // because it does not produce sorted output
    let (final_perm, merge_depth) = if is_top_level {
        alekseev_merge(builder, &v_perm, &w_perm, k, merge_start_layer)
    } else {
        build_recursive_merge(builder, v_perm, w_perm, k, merge_start_layer)
    };

    (final_perm, recursive_depth + merge_depth)
}

/// Main entry point: Implements `BatcherNetwork` initialization logic.
///
/// `n`: Total number of input wires.
/// `k`: The number of smallest elements to find.
#[cached(size = 200, result = true)]
pub fn min_k_batcher_sort_network(n: usize, k: usize) -> Result<SwapNetwork> {
    if n < k {
        bail!("Cannot construct min-k selection network for n < k; n = {n}, k = {k}");
    }

    let mut builder = MinKBatcherBuilder::new();
    let initial_indices: Vec<usize> = (0..n).collect();

    let mut perm = if k <= n - k {
        // Standard Min(k) network construction
        let (perm, _) = build_recursive_sort(
            &mut builder,
            initial_indices,
            k,
            0,
            true, // is_top_level
        );
        perm
    } else {
        // Optimization for large K: Build Max(N-K) then invert
        let target_k = n - k;
        let (perm, _) =
            build_recursive_sort(&mut builder, initial_indices.clone(), target_k, 0, true);

        // Invert comparators: (a, b) -> (b, a)
        // In standard sorting networks, (a, b) sorts such that a < b.
        // Swapping them to (b, a) sorts such that b < a (descending), effectively making it a Max filter.
        for layer_comps in builder.comparators_by_layer.values_mut() {
            let swapped: HashSet<(usize, usize)> =
                layer_comps.iter().map(|(a, b)| (*b, *a)).collect();
            *layer_comps = swapped;
        }

        let perm_set: HashSet<usize> = perm.iter().cloned().collect();
        let complement: Vec<usize> = (0..n).filter(|i| !perm_set.contains(i)).collect();
        complement
    };

    // The recursive network-building procedure also returns a list of indices
    // which will hold the expected results after the swaps are executed.
    // In general, this list is not a prefix of the original sequence.
    // To make it a prefix, we map the network with the inverse permutation
    let mut swap_network = builder.to_swap_network();

    // At this point, `perm` contains at most `k` elements, we complete it
    // with missing indices, but their order is irelevant.
    for i in 0..n {
        if !perm.contains(&i) {
            perm.push(i);
        }
    }

    // Compute inverse permutation.
    let mut inv_perm = vec![0; n];
    for (i, &idx) in perm.iter().enumerate() {
        assert!(idx < n);
        inv_perm[idx] = i;
    }

    // Apply inverse permutation on the swap network indices.
    swap_network.map_indices(|i| {
        inv_perm
            .get(i)
            .copied()
            .ok_or_eyre("Unexpected: permutation index out of bounds")
    })?;
    // Now the Min-K elements form a prefix of the sequence.

    Ok(swap_network)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    // Values from Python experiments:
    // https://www.notion.so/inversed/Top-K-selection-networks-288a3b540bfe80b7b05fe98b7cae7a89a

    fn test_network_size() {
        for (n, k, expected_depth, expected_comps) in [
            (512, 1, 9, 511),
            (512, 4, 22, 1652),
            (512, 64, 36, 5944),
            (512, 256, 37, 7934),
            (768, 320, 46, 13277),
            (500, 320, 37, 7374),
            (350, 320, 34, 3054),
        ] {
            let network = min_k_batcher_sort_network(n, k).unwrap();
            assert_eq!(network.num_layers(), expected_depth);
            assert_eq!(network.num_comparisons(), expected_comps);
        }
    }

    #[test]
    fn test_randomized_inputs() {
        let mut rng = rand::thread_rng();
        let n_tests = 1000;

        for _ in 0..n_tests {
            let n = rng.gen_range(1..20);
            let k = rng.gen_range(1..=n);

            let network = min_k_batcher_sort_network(n, k).unwrap();

            let mut input: Vec<i32> = (0..n).map(|_| rng.gen_range(0..30)).collect();

            let mut expected_top_k = input.clone();
            expected_top_k.sort();
            expected_top_k.truncate(k);

            network.apply(&mut input);
            let mut output = input;
            output.truncate(k);
            output.sort();

            assert_eq!(output, expected_top_k, "Output must preserve elements");
        }
    }
}
