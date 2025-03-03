//! Implements a sorting network which identifies the minimum element of an
//! unsorted list in logarithmically many rounds of parallel comparisons using
//! a binary tree design.

use super::sorting_network::{SortingNetwork, SortingNetworkLayer};

/// Generates a sorting network which moves the minimum element of an unsorted
/// list of size `length` to the starting index in logarithmically many rounds
/// using a binary tree of comparison operations.
pub fn tree_min(length: usize) -> SortingNetwork {
    match length {
        0 | 1 => SortingNetwork::new(),
        _ => {
            let deg = (usize::ilog2(length - 1) + 1) as usize;
            let mut network = tree_min_base(deg);
            network.filter_wires(|&(_, idx2)| idx2 < length);
            network
        }
    }
}

/// Generates a `min` sorting network for a list of length `2^deg`.
fn tree_min_base(deg: usize) -> SortingNetwork {
    SortingNetwork {
        layers: (0..deg).map(|stage| tree_min_layer(deg, stage)).collect(),
    }
}

/// Generates a sorting network layer with wires between every other multiple of
/// `2^stage` indices, for indices up to `2^deg` exclusive.
fn tree_min_layer(deg: usize, stage: usize) -> SortingNetworkLayer {
    if stage >= deg {
        return Default::default();
    }

    // Here 0 <= stage < deg
    let (increment, count) = (1 << stage, 1 << (deg - stage - 1));
    (0..count)
        .map(|n_wire| (2 * n_wire * increment, (2 * n_wire + 1) * increment))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn check_small_tree_min_networks() {
        // (length, network)
        let hardcoded_min_trees = [
            (0, vec![]),
            (1, vec![]),
            (2, vec![vec![(0, 1)]]),
            (3, vec![vec![(0, 1)], vec![(0, 2)]]),
            (4, vec![vec![(0, 1), (2, 3)], vec![(0, 2)]]),
            (6, vec![vec![(0, 1), (2, 3), (4, 5)], vec![(0, 2)], vec![(
                0, 4,
            )]]),
            (8, vec![
                vec![(0, 1), (2, 3), (4, 5), (6, 7)],
                vec![(0, 2), (4, 6)],
                vec![(0, 4)],
            ]),
            (13, vec![
                vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
                vec![(0, 2), (4, 6), (8, 10)],
                vec![(0, 4), (8, 12)],
                vec![(0, 8)],
            ]),
            (15, vec![
                vec![(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)],
                vec![(0, 2), (4, 6), (8, 10), (12, 14)],
                vec![(0, 4), (8, 12)],
                vec![(0, 8)],
            ]),
        ];

        for (length, hardcoded_network) in hardcoded_min_trees.iter() {
            assert_eq!(tree_min(*length).layers, *hardcoded_network);
        }
    }

    #[test]
    fn test_tree_min_execution() {
        let mut rng = rand::thread_rng();

        for _ in 0..10 {
            let length = rng.gen_range(16..1024);
            let network = tree_min(length);

            for _ in 0..20 {
                let mut vals: Vec<u64> = (0..length).map(|_| rng.gen_range(0..100)).collect();

                let seq_min = *(vals.iter().min().unwrap());

                network.apply(&mut vals);
                let network_min = vals[0];

                assert_eq!(seq_min, network_min);
            }
        }
    }
}
