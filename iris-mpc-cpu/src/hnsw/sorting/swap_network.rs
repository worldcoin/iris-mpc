use crate::{
    hawkers::aby3::aby3_store::{Aby3DistanceRef, Aby3Store},
    hnsw::VectorStore,
    shares::Share,
};
use eyre::{eyre, Result};
use itertools::Itertools;

/// Type of a single layer in a non-adaptive comparator network represented by
/// the `SwapNetwork` struct.
pub type SwapNetworkLayer = Vec<(usize, usize)>;

/// Struct representing a non-adaptive comparator network, here called a "swap"
/// network.
///
/// Networks are represented as a list of lists of usize tuples. The exterior
/// list represents sequential steps of parallel comparisons in the network,
/// while the interior lists represent the wires of the network in the
/// associated step.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SwapNetwork {
    pub layers: Vec<SwapNetworkLayer>,
}
impl SwapNetwork {
    /// Create a new empty swap network.
    pub fn new() -> Self {
        Default::default()
    }

    /// Returns the number of sequential layers in the network
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total number of comparison wires in the network
    pub fn num_comparisons(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }

    /// Apply a map to the indices of the swap network.
    pub fn map_indices<F>(&mut self, map: F) -> Result<&mut Self>
    where
        F: Fn(usize) -> Result<usize>,
    {
        for layer in self.layers.iter_mut() {
            for wire in layer.iter_mut() {
                let (idx1, idx2) = *wire;
                *wire = (map(idx1)?, map(idx2)?);
            }
        }
        Ok(self)
    }

    /// Uniformly shift indices of an input swap network. Returns an error if integer
    /// overflow occurs during a shift operation.
    pub fn shift(&mut self, shift_amount: isize) -> Result<&mut Self> {
        self.map_indices(|x| {
            x.checked_add_signed(shift_amount)
                .ok_or(eyre!("Integer overflow due to shifting"))
        })
    }

    /// Apply a filter to wires of the swap network, removing any layers
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

    /// Apply this swap network to an input array slice of elements with a
    /// totally ordered type.
    pub fn apply<F: Ord>(&self, list: &mut [F]) {
        self.layers
            .iter()
            .for_each(|layer| SwapNetwork::apply_layer(layer, list));
    }

    /// Apply a single swap network layer to an intput array slice of
    /// elements with a totally ordered type.
    pub fn apply_layer<F: Ord>(layer: &SwapNetworkLayer, list: &mut [F]) {
        layer.iter().for_each(|(idx1, idx2)| {
            if let (Some(val1), Some(val2)) = (list.get(*idx1), list.get(*idx2)) {
                if val1 > val2 {
                    list.swap(*idx1, *idx2);
                }
            }
        })
    }

    /// Combine two swap networks in parallel by merging layers in sequence.
    pub fn merge_parallel(n1: SwapNetwork, n2: SwapNetwork) -> SwapNetwork {
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

        SwapNetwork { layers }
    }

    /// Combine two swap networks in series by concatenating network layers.
    pub fn merge_series(n1: SwapNetwork, n2: SwapNetwork) -> SwapNetwork {
        let layers = n1.layers.into_iter().chain(n2.layers).collect();

        SwapNetwork { layers }
    }
}

/// Function applies the supplied swap network `network` to the list of
/// tuples `(VectorRef, DistanceRef)` using a given `VectorStore` object to
/// execute comparisons for each layer in batches.
pub async fn apply_swap_network<V: VectorStore>(
    store: &mut V,
    list: &mut [(V::VectorRef, V::DistanceRef)],
    network: &SwapNetwork,
) -> Result<()> {
    for layer in network.layers.iter() {
        let distances: Vec<_> = layer
            .iter()
            .filter_map(
                |(idx1, idx2): &(usize, usize)| match (list.get(*idx1), list.get(*idx2)) {
                    // swap order to check for strict inequality d1 > d2
                    (Some((_, d1)), Some((_, d2))) => Some((d2.clone(), d1.clone())),
                    _ => None,
                },
            )
            .collect();
        let comp_results = store.less_than_batch(&distances).await?;
        for ((idx1, idx2), is_gt) in layer.iter().zip(comp_results) {
            if is_gt {
                list.swap(*idx1, *idx2)
            }
        }
    }

    Ok(())
}

/// Function obliviously applies the supplied swap network `network` to the list of
/// tuples containing ids and distances between iris vectors as `(u32, Aby3DistanceRef)`.
/// An 'Aby3Store' object executes comparisons via MPC for each layer in batches.
/// Note that output is secret-shared, even for unchanged elements of the list.
/// This implies that all vector ids are considered secret-shared after the first layer,
/// which might introduce an additional throughput overhead.
/// For example, for a swap network implementing the tournament method to find the minimum of a list of length N,
/// this throughput overhead is O(1).
pub async fn apply_oblivious_swap_network(
    store: &mut Aby3Store,
    list: &[(u32, Aby3DistanceRef)],
    network: &SwapNetwork,
) -> Result<Vec<(Share<u32>, Aby3DistanceRef)>> {
    let mut encrypted_list = vec![];
    for (layer_id, layer) in network.layers.iter().enumerate() {
        let distances: Vec<_> = layer
            .iter()
            .filter_map(
                |(idx1, idx2): &(usize, usize)| match (list.get(*idx1), list.get(*idx2)) {
                    // swap order to check for strict inequality d1 > d2
                    (Some((_, d1)), Some((_, d2))) => Some((d2.clone(), d1.clone())),
                    _ => None,
                },
            )
            .collect();
        // Computes d1 > d2 without opening as in less_than_batch
        let comp_results = store.oblivious_less_than_batch(&distances).await?;
        encrypted_list = if layer_id == 0 {
            // First layer: input ids are in plaintext, so we can use the more efficient plain_ids version.
            store
                .oblivious_swap_batch_plain_ids(comp_results, list, layer)
                .await?
        } else {
            // Following layers: input ids are secret shared
            store
                .oblivious_swap_batch(comp_results, &encrypted_list, layer)
                .await?
        };
    }

    Ok(encrypted_list)
}
