//! Implementation of Batcher Odd-even Sorting Networks, including a custom
//! extension supporting generation of optimized networks for non-2-power lists
//! with a partially sorted prefix of elements, as required for batch insertion
//! of new elements into a sorted list. For more details on the standard
//! Batcher sorting network, see references:
//!
//! - (<https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort>)
//! - (<https://math.mit.edu/~shor/18.310/batcher.pdf>)

use eyre::Result;

use super::swap_network::{SwapNetwork, SwapNetworkLayer};

/// Generate swap network wire tuples for stage `stage` of the batcher merge
/// network of degree `deg`.
fn batcher_merge_step(deg: usize, stage: usize) -> SwapNetworkLayer {
    // Uniform distance between wire indices of this merge layer
    let offset = 1 << (deg - stage);

    if stage == 1 {
        (0..offset).map(|start| (start, start + offset)).collect()
    } else {
        // Number of consecutive wire clusters of size `offset` in this merge layer
        let scale = (1 << (stage - 1)) - 1;

        let mut sort_tuples = Vec::with_capacity(offset * scale);
        for start in 0..offset {
            sort_tuples.extend(
                (1..=scale).map(|k| (start + (2 * k - 1) * offset, start + (2 * k) * offset)),
            )
        }
        sort_tuples
    }
}

/// Generate the merging network for Batcher Odd-even Merge Sort of degree
/// `deg`. The merge network of degree `deg` consists of `deg` simple layers,
/// and produces a sorted list of length `2^k` when applied to a list of this
/// length whose first and second halves are already individually sorted.
fn batcher_merger_network(deg: usize) -> SwapNetwork {
    let layers = (1..=deg)
        .map(|stage| batcher_merge_step(deg, stage))
        .collect();
    SwapNetwork { layers }
}

/// Generate the pruned Batcher odd-even merge sort sorting network for an input
/// list of a specified size. A "pruned" network is the network obtained by
///
/// - starting with a full 2-power size batcher network of minimal size needed
///   to sort the targeted potentially non-2-power size
/// - removing any wires comparing an in-range index with a (larger)
///   out-of-range index
///
/// Conceptually, the out-of-range indices can be thought of as indices with
/// values initialized to "infinity", so any comparison with an element in the
/// proper list acts as a no-op.
///
/// For 2-power sizes, the network produced by this function is the standard
/// Batcher odd-even merge sort network.
pub fn batcher_network(size: usize) -> Result<SwapNetwork> {
    partial_batcher_network(0, size)
}

/// Generate a pruned Batcher odd-even merge sort network which sorts an input
/// list where a prefix of the elements already appear in sorted order.
///
/// Two rules are applied to a basic 2-power size batcher network to eliminate
/// unnecessary wires from the network:
///
/// - Prefix networks used to sort the input halves of the batcher merger
///   network are omitted if the inputs are already in sorted order
/// - Wires comparing with trailing indices that are "stable" -- meaning either
///   the index is out of bounds, or belongs to a tail of elements already
///   determined to be maximal and sorted -- are omitted from the network
///
/// For the latter rule, a somewhat nuanced observation is used to identify
/// extra indices that are "stabilized early" coming from the assumption that
/// some portion of the prefix is already sorted. Recall that the batcher
/// sorting network begins by sorting two halves of its input, then combining
/// the results with an efficient "merger" network. Using such an approach, if
/// the first half is already sorted, and the second half begins with some
/// additional number of sorted elements that are the "top part" of the sorted
/// region of the input, then after sorting the second half, this number of
/// previously sorted elements is known already to be larger than all elements
/// of the first half, and so are sorted into their final positions prior to
/// applying the merger network. As a result, merger wires doing additional
/// comparisons with these indices can be omitted from the final network.
///
/// Indices which fall into this category are identified during the recursive
/// network construction provided by the `batcher_recursive` function call. See
/// documentation there for more details.
pub fn partial_batcher_network(
    sorted_prefix_size: usize,
    unsorted_size: usize,
) -> Result<SwapNetwork> {
    let total_list_size = sorted_prefix_size + unsorted_size;
    assert_ne!(total_list_size, 0);
    let deg = match total_list_size {
        1 => 0,
        _ => (usize::ilog2(total_list_size - 1) + 1) as usize,
    };

    batcher_recursive(deg, 0, sorted_prefix_size, total_list_size)
}

/// Recursively construct the partial batcher sorting network for a list with a
/// pre-sorted prefix of elements, where:
///
/// - The active window of indices is the half-open interval `[ offset*2^deg,
///   (offset+1)*2^deg )`
/// - `unsorted_idx` specifies the starting index of the unsorted region
/// - `stable_idx` specifies the starting index of the stable region,
///   representing elements which should not be considered by the final network
///   (e.g. indices beyond the end of the list to be sorted)
///
/// For an active window that is entirely contained in either the sorted region
/// or the stable region of the list, this returns an empty sorting network.
/// Otherwise, networks are recursively generated for each half of the active
/// window, and are merged using a suitably pruned Batcher merging network.
///
/// To identify elements which are "stable" for the purposes of this merging
/// network, the function identifies the number of pre-sorted elements which
/// are included in the second half's window, and adds to this the number of
/// indices which are stable by way of falling after the specified `stable_idx`
/// value. This total represents the number of indices at the end of the
/// current active window which can be ignored (all connecting wires filtered
/// out) in the merging network.
pub fn batcher_recursive(
    deg: usize,
    offset: usize,
    unsorted_idx: usize,
    stable_idx: usize,
) -> Result<SwapNetwork> {
    assert!(unsorted_idx <= stable_idx);

    // Parameters of active index window
    let window_size = 1 << deg;
    let start_idx = offset * window_size; // inclusive
    let end_idx = (offset + 1) * window_size; // exclusive

    if end_idx <= unsorted_idx || start_idx >= stable_idx || deg == 0 {
        // Cases where no sorting occurs. Note that this frequently handles
        // larger values of deg without requiring additional levels of recursion.
        Ok(Default::default())
    } else {
        let prefix_1 = batcher_recursive(deg - 1, 2 * offset, unsorted_idx, stable_idx)?;
        let prefix_2 = batcher_recursive(deg - 1, 2 * offset + 1, unsorted_idx, stable_idx)?;
        let prefix = SwapNetwork::merge_parallel(prefix_1, prefix_2);

        // Compute how many trailing indices can be ignored in the merging network
        let window_n_stable = end_idx.saturating_sub(stable_idx);
        let mid_idx = (start_idx + end_idx) / 2;
        let prefix_2_n_sorted = unsorted_idx.saturating_sub(mid_idx);
        let total_stable_amount = window_n_stable + prefix_2_n_sorted;

        let mut merger = batcher_merger_network(deg);
        merger
            .shift(start_idx as isize)?
            .filter_wires(|(_, idx2)| *idx2 < end_idx - total_stable_amount);

        Ok(SwapNetwork::merge_series(prefix, merger))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn check_small_batcher_networks() -> Result<()> {
        let hardcoded_batchers = [
            vec![],
            vec![vec![(0usize, 1usize)]],
            vec![vec![(0, 1), (2, 3)], vec![(0, 2), (1, 3)], vec![(1, 2)]],
            vec![
                vec![(0, 1), (2, 3), (4, 5), (6, 7)],
                vec![(0, 2), (1, 3), (4, 6), (5, 7)],
                vec![(1, 2), (5, 6)],
                vec![(0, 4), (1, 5), (2, 6), (3, 7)],
                vec![(2, 4), (3, 5)],
                vec![(1, 2), (3, 4), (5, 6)],
            ],
        ];

        for (deg, hardcoded_network) in hardcoded_batchers.iter().enumerate() {
            let size = 1usize << deg;
            assert_eq!(batcher_network(size)?.layers, *hardcoded_network);
        }

        Ok(())
    }

    #[test]
    fn test_full_batcher_sorting() -> Result<()> {
        let mut rng = rand::thread_rng();

        for deg in 0..6 {
            let size = 1usize << deg;
            let network = batcher_network(size)?;

            for _ in 0..50 {
                let mut vals1: Vec<u64> = (0..size).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }

        Ok(())
    }

    #[test]
    fn test_batcher_arbitrary_length() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let length = rng.gen_range(16..1024);
            let network = partial_batcher_network(0, length)?;

            for _ in 0..20 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }
        Ok(())
    }

    #[test]
    fn test_batcher_insertion() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let sorted_length = rng.gen_range(128..512);
            let unsorted_length = rng.gen_range(128..512);
            let length = sorted_length + unsorted_length;
            let network = partial_batcher_network(sorted_length, unsorted_length)?;

            for _ in 0..20 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                vals1.get_mut(0..sorted_length).unwrap().sort();
                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }

        Ok(())
    }
}
