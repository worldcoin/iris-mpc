use itertools::Itertools;

/// Type of a single layer in a non-adaptive sorting network represented by the
/// `SortingNetwork` struct.
pub type SortingNetworkLayer = Vec<(usize, usize)>;

/// Struct representing a non-adaptive sorting network.
///
/// Networks are represented as a list of lists of usize tuples.  The exterior
/// list represents sequential steps of parallel comparisons in the network,
/// while the interior lists represent the wires of the network in the
/// associated step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SortingNetwork {
    pub layers: Vec<SortingNetworkLayer>,
}
impl SortingNetwork {
    /// Create a new empty sorting network.
    pub fn new() -> Self {
        Default::default()
    }

    /// Apply a map to the indices of the sorting network.
    pub fn map_indices<F>(&mut self, map: F) -> &mut Self
    where
        F: Fn(usize) -> usize,
    {
        self.layers.iter_mut().for_each(|layer| {
            layer.iter_mut().for_each(|wire| {
                let (idx1, idx2) = wire;
                *wire = (map(*idx1), map(*idx2))
            })
        });
        self
    }

    /// Uniformly shift indices of an input sorting network.  Panics if integer
    /// overflow occurs during a shift operation.
    pub fn shift(&mut self, shift_amount: isize) -> &mut Self {
        self.map_indices(|x| x.checked_add_signed(shift_amount).unwrap())
    }

    /// Apply a filter to wires of the sorting network, removing any layers
    /// which are empty in the output.
    pub fn filter_wires<F>(&mut self, predicate: F) -> &mut Self
    where
        F: Fn(&(usize, usize)) -> bool,
    {
        self.layers
            .iter_mut()
            .for_each(|layer| layer.retain(&predicate));
        self.layers.retain(|layer: &Vec<_>| !layer.is_empty());
        self
    }

    /// Apply this sorting network to an input array slice of elements with a
    /// totally ordered type.
    pub fn apply<F: Ord>(&self, list: &mut [F]) {
        self.layers
            .iter()
            .for_each(|layer| SortingNetwork::apply_layer(layer, list));
    }

    /// Apply a single sorting network layer to an intput array slice of
    /// elements with a totally ordered type.
    pub fn apply_layer<F: Ord>(layer: &SortingNetworkLayer, list: &mut [F]) {
        layer.iter().for_each(|(idx1, idx2)| {
            if let (Some(val1), Some(val2)) = (list.get(*idx1), list.get(*idx2)) {
                if val1 > val2 {
                    list.swap(*idx1, *idx2);
                }
            }
        })
    }

    /// Combine two sorting networks in parallel by merging layers in sequence.
    pub fn merge_parallel(n1: SortingNetwork, n2: SortingNetwork) -> SortingNetwork {
        let layers = n1
            .layers
            .into_iter()
            .zip_longest(n2.layers)
            .map(|layers| match layers {
                itertools::EitherOrBoth::Left(l1) => l1,
                itertools::EitherOrBoth::Right(l2) => l2,
                itertools::EitherOrBoth::Both(l1, l2) => l1.into_iter().chain(l2).collect(),
            })
            .collect();

        SortingNetwork { layers }
    }

    /// Combine two sorting networks in series by concatenating network layers.
    pub fn merge_series(n1: SortingNetwork, n2: SortingNetwork) -> SortingNetwork {
        let layers = n1.layers.into_iter().chain(n2.layers).collect();

        SortingNetwork { layers }
    }
}
