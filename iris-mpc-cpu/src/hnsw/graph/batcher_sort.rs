/// Implementation of sorting networks  for batcher odd-even merge sort.
///
/// Representation:
/// - Sorting network is a list of lists of usize tuples
/// - Exterior list represents sequential steps of parallel comparisons
/// - Interior list represents the comparison pairs of the network in the
///   associated step.

use itertools::Itertools;

pub type SortingNetworkLayer = Vec<(usize, usize)>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SortingNetwork {
    pub layers: Vec<SortingNetworkLayer>,
}

impl SortingNetwork {

    pub fn new() -> Self {
        Default::default()
    }

    /// Apply a map to the indices of the sorting network
    pub fn map_indices<F>(&mut self, map: F) -> &mut Self
    where
        F: Fn(usize) -> usize,
    {
        self.layers.iter_mut()
            .for_each(|layer| {
                layer
                    .iter_mut()
                    .for_each(|wire| {
                        let (idx1, idx2) = wire;
                        *wire = (map(*idx1), map(*idx2))
                    })
            });
        self
    }

    /// Uniformly shift indices of an input sorting network.  Panics if integer
    /// overflow occurs during a shift operation.
    pub fn shift(&mut self, shift_amount: isize) -> &mut Self {
        self.map_indices(|x| { x.checked_add_signed(shift_amount).unwrap() })
    }

    /// Apply a filter to wires of the sorting network, optionally removing any
    /// layers which are empty in the output
    pub fn filter_wires<F>(&mut self, predicate: F, purge_empty: bool) -> &mut Self
    where
        F: Fn(&(usize, usize)) -> bool,
    {
        self.layers.iter_mut().for_each(|layer| layer.retain(&predicate));
        if purge_empty {
            self.layers.retain(|layer: &Vec<_>| !layer.is_empty());
        }
        self
    }

    /// Apply this sorting network to an input array slice of ordered elements
    pub fn apply<F: Ord>(&self, list: &mut [F]) {
        self.layers.iter().for_each(|layer| SortingNetwork::apply_layer(layer, list));
    }

    pub fn apply_layer<F: Ord>(layer: &SortingNetworkLayer, list: &mut [F]) {
        layer.iter().for_each(|(idx1, idx2)| {
            if let (Some(val1), Some(val2)) = (list.get(*idx1), list.get(*idx2)) {
                if val1 > val2 {
                    list.swap(*idx1, *idx2);
                }
            }
        })
    }

    pub fn merge_parallel(n1: SortingNetwork, n2: SortingNetwork) -> SortingNetwork {
        let layers = n1.layers.into_iter()
            .zip_longest(n2.layers.into_iter())
            .map(|layers| match layers {
                itertools::EitherOrBoth::Left(l1) => l1,
                itertools::EitherOrBoth::Right(l2) => l2,
                itertools::EitherOrBoth::Both(l1, l2) => l1.into_iter().chain(l2.into_iter()).collect(),
            })
            .collect();

        SortingNetwork { layers }
    }

    /// Combine two sorting networks in series
    pub fn merge_series(n1: SortingNetwork, n2: SortingNetwork) -> SortingNetwork {
        let layers = n1.layers.into_iter().chain(n2.layers).collect();

        SortingNetwork { layers }
    }
}

/// Generate sort tuples for specified stage of batcher merge network
pub fn batcher_merge_step(deg: usize, stage: usize) -> SortingNetworkLayer {
    let (offset, scale) = (1 << deg - stage, 1 << stage - 1);
    let n_comps = if stage == 1 {
        offset
    } else {
        offset * (scale - 1)
    };

    let mut sort_tuples = Vec::with_capacity(n_comps);
    for start in 0..offset {
        if stage == 1 {
            sort_tuples.push((start, start + offset));
        } else {
            sort_tuples.extend(
                (1..=(scale - 1)).map(|k| (start + (2 * k - 1) * offset, start + (2 * k) * offset)),
            )
        }
    }

    sort_tuples
}

pub fn batcher_merge_network(deg: usize) -> SortingNetwork {
    let layers = (1..=deg)
        .map(|stage| batcher_merge_step(deg, stage))
        .collect();
    SortingNetwork { layers }
}

/// Generate full Batcher odd-even merge sort sorting network for 2-power size
pub fn batcher_full_network(deg: usize) -> SortingNetwork {
    match deg {
        0 => Default::default(),
        1 => batcher_merge_network(1),
        _ => {
            let prefix_1 = batcher_full_network(deg - 1);
            let mut prefix_2 = prefix_1.clone();
            prefix_2.shift(1 << (deg - 1));
            let prefix = SortingNetwork::merge_parallel(prefix_1, prefix_2);
            let merger = batcher_merge_network(deg);
            SortingNetwork::merge_series(prefix, merger)
        }
    }
}

pub fn batcher_partial_network(sorted_prefix_size: usize, unsorted_size: usize) -> SortingNetwork {
    let total_list_size = sorted_prefix_size + unsorted_size;
    let deg = (usize::ilog2(total_list_size - 1) + 1) as usize;

    let network = batcher_general(deg, 0, sorted_prefix_size, total_list_size);
    network
}

pub fn batcher_general(
    deg: usize,
    offset: usize,
    unsorted_idx: usize,
    stable_idx: usize,
) -> SortingNetwork {
    assert!(unsorted_idx <= stable_idx);
    let size = 1 << deg;
    let start_idx = offset * size; // inclusive
    let end_idx = (offset + 1) * size; // exclusive

    if end_idx <= unsorted_idx || start_idx >= stable_idx || deg == 0 {
        Default::default()
    } else {
        let prefix_1 = batcher_general(deg - 1, 2 * offset, unsorted_idx, stable_idx);
        let prefix_2 = batcher_general(deg - 1, 2 * offset + 1, unsorted_idx, stable_idx);
        let prefix = SortingNetwork::merge_parallel(prefix_1, prefix_2);

        let window_stable_suffix_size = end_idx.saturating_sub(stable_idx);

        let mid_idx = (start_idx + end_idx) / 2;
        let prefix_2_n_sorted = unsorted_idx.saturating_sub(mid_idx);

        let total_stable_amount = window_stable_suffix_size + prefix_2_n_sorted;

        let mut merger = batcher_merge_network(deg);
        merger.shift(start_idx as isize).filter_wires(|(_, idx2)| *idx2 < end_idx - total_stable_amount, true);
        // merger = shift_network(merger, start_idx);
        // merger = filter_network(merger, |(_, idx2)| *idx2 < end_idx - total_stable_amount);
        let network = SortingNetwork::merge_series(prefix, merger);

        network
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn check_small_batcher_networks() {
        let hardcoded_batchers = vec![
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

        for deg in 0..hardcoded_batchers.len() {
            assert_eq!(batcher_full_network(deg).layers, hardcoded_batchers[deg]);
        }
    }

    #[test]
    fn test_full_batcher_sorting() {
        let mut rng = rand::thread_rng();

        for deg in 0..6 {
            let network = batcher_full_network(deg);
            let length = 1usize << deg;

            for _ in 0..50 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }
    }

    #[test]
    fn test_batcher_arbitrary_length() {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let length = rng.gen_range(16..1024);
            let network = batcher_partial_network(0, length);

            for _ in 0..20 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }
    }

    #[test]
    fn test_batcher_insertion() {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let sorted_length = rng.gen_range(128..512);
            let unsorted_length = rng.gen_range(128..512);
            let length = sorted_length + unsorted_length;
            let network = batcher_partial_network(sorted_length, unsorted_length);

            for _ in 0..20 {
                let mut vals1: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();
                let mut vals2 = vals1.clone();

                vals1.get_mut(0..sorted_length).unwrap().sort();
                network.apply(&mut vals1);
                vals2.sort();

                assert_eq!(vals1, vals2);
            }
        }
    }
}
