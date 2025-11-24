//! Implementation of HNSW algorithm for k-nearest-neighbor search over iris
//! biometric templates with high-latency MPC comparison operations. Based on
//! the `HawkSearcher` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use super::{
    graph::neighborhood::SortedNeighborhoodV,
    sorting::{
        quickselect::run_quickselect_with_store, swap_network::apply_swap_network,
        tree_min::tree_min,
    },
    vector_store::VectorStoreMut,
};
use crate::hnsw::{metrics::ops_counter::Operation, SortedNeighborhood, VectorStore};

use crate::hnsw::GraphMem;

use aes_prng::AesRng;
use ampc_actor_utils::fast_metrics::FastHistogram;
use eyre::{bail, eyre, OptionExt, Result};
use itertools::{izip, Itertools};
use rand::{RngCore, SeedableRng};
use rand_distr::{Distribution, Geometric};
use serde::{Deserialize, Serialize};
use siphasher::sip::SipHasher13;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{Hash, Hasher},
    iter::once,
};
use tracing::{debug, instrument, trace_span, Instrument};

/// The number of explicitly provided parameters for different layers of HNSW
/// search, used by the `HnswParams` struct.
pub const N_PARAM_LAYERS: usize = 5;
const M_MAX_MULTIPLIER: f64 = 1.0;

/// Struct specifying general parameters for HNSW search.
///
/// Most algorithm and graph properties are specified per-layer by a
/// fixed-length array of size `N_PARAM_LAYERS`, for layers up to
/// `N_PARAM_LAYERS - 1`. Layers larger than this use the last set of
/// parameters.
#[allow(non_snake_case)]
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct HnswParams {
    /// The number of neighbors for insertion
    pub M: [usize; N_PARAM_LAYERS],

    /// Target maximum number of neighbors allowed per graph node
    pub M_max: [usize; N_PARAM_LAYERS],

    /// Limit number of neighbors allowed per graph node; any neighborhood
    /// larger than this size should be compacted to size at most `M_max`.
    pub M_limit: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` for search layers during construction
    pub ef_constr_search: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` for insertion layers during construction
    pub ef_constr_insert: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` during search
    pub ef_search: [usize; N_PARAM_LAYERS],
}

#[allow(non_snake_case)]
impl HnswParams {
    /// Construct a `Params` object corresponding to parameter configuration providing the
    /// functionality described in the original HNSW paper of Malkov and Yashunin:
    /// - `ef_construction` exploration factor used for insertion layers
    /// - `ef_search` exploration factor used for layer 0 in search
    /// - higher layers in both insertion and search use exploration factor 1,
    ///   representing simple greedy search
    /// - vertex degrees bounded by `M_max = M` in positive layer graphs
    /// - vertex degrees bounded by `M_max0 = 2*M` in layer 0 graph
    /// - `m_L = 1 / ln(M)` so that layer density decreases by a factor of `M` at
    ///   each successive hierarchical layer
    pub fn new(ef_construction: usize, ef_search: usize, M: usize) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let M_limit_arr = M_max_arr.map(|m| (m as f64 * M_MAX_MULTIPLIER) as usize);
        let ef_constr_search_arr = [1usize; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef_construction; N_PARAM_LAYERS];
        let mut ef_search_arr = [1usize; N_PARAM_LAYERS];
        ef_search_arr[0] = ef_search;

        Self {
            M: M_arr,
            M_max: M_max_arr,
            M_limit: M_limit_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
        }
    }

    /// Parameter configuration using fixed exploration factor for all layer
    /// search operations, both for insertion and for search.
    pub fn new_uniform(ef: usize, M: usize) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let M_limit_arr = M_max_arr.map(|m| (m as f64 * M_MAX_MULTIPLIER) as usize);
        let ef_constr_search_arr = [ef; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef; N_PARAM_LAYERS];
        let ef_search_arr = [ef; N_PARAM_LAYERS];

        Self {
            M: M_arr,
            M_max: M_max_arr,
            M_limit: M_limit_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
        }
    }

    pub fn get_M(&self, lc: usize) -> usize {
        Self::get_val(&self.M, lc)
    }

    pub fn get_M_max(&self, lc: usize) -> usize {
        Self::get_val(&self.M_max, lc)
    }

    pub fn get_M_limit(&self, lc: usize) -> usize {
        Self::get_val(&self.M_limit, lc)
    }

    pub fn get_ef_constr_search(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_constr_search, lc)
    }

    pub fn get_ef_constr_insert(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_constr_insert, lc)
    }

    pub fn get_ef_search(&self, lc: usize) -> usize {
        Self::get_val(&self.ef_search, lc)
    }

    #[inline(always)]
    /// Select value at index `lc` from the input fixed-size array, or the last
    /// index of this array if `lc` is larger than the array size.
    fn get_val(arr: &[usize; N_PARAM_LAYERS], lc: usize) -> usize {
        arr[lc.min(N_PARAM_LAYERS - 1)]
    }
}

/// Struct specifies how layers are handled by an `HnswSearcher`.`
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub enum LayerMode {
    /// Standard operation: maintains single entry point and updates it when a
    /// new node is inserted as the first item in a new highest layer.
    ///
    /// Graph search starts at the unique entry point.
    Standard,

    /// Bounded standard operation: maintains a single entry point and updates
    /// it when a new node is inserted as the first item in a new highest layer.
    /// Node insertion is bounded at a fixed maximum layer height.
    ///
    /// Graph search starts at the unique entry point.
    Bounded {
        /// Maximum layer for node insertion
        max_graph_layer: usize,
    },

    /// Nodes are inserted at up to a maximum layer height, and any node which
    /// would be inserted at a higher layer than this is added to an ongoing
    /// list of entry points.
    ///
    /// Graph search starts in the top layer, at a node of minimal distance from
    /// the query among the entry points, calculated by a direct linear scan of
    /// the entry points list.
    LinearScan {
        /// Maximum layer for node insertion
        max_graph_layer: usize,
    },
}

/// Struct specifies the probability distribution used to generate insertion
/// layers for new nodes in an HNSW graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LayerDistribution {
    Geometric {
        /// Probability `q = 1-p` for geometric distribution of layer densities
        layer_probability: f64,
    },
}

#[allow(non_snake_case)]
impl LayerDistribution {
    pub fn new_geometric_from_M(M: usize) -> Self {
        let layer_probability = (M as f64).recip();
        LayerDistribution::Geometric { layer_probability }
    }

    /// Generate a random layer based on the specified `LayerDistribution` enum
    /// variant.
    pub fn gen_layer(&self, rng: &mut impl RngCore) -> Result<usize> {
        match self {
            LayerDistribution::Geometric { layer_probability } => {
                let p_geom = 1f64 - layer_probability;
                let geom_distr = Geometric::new(p_geom)?;

                let layer = geom_distr.sample(rng) as usize;
                Ok(layer)
            }
        }
    }

    /// Compute the parameter m_L associated with a geometric distribution
    /// parameter q describing the random layer of newly inserted graph nodes.
    ///
    /// E.g. for graph hierarchy where each layer has a factor of 32 fewer
    /// entries than the last, the `layer_probability` input is 1/32.
    pub fn m_L_from_layer_probability(layer_probability: f64) -> f64 {
        -layer_probability.ln().recip()
    }

    /// Compute the parameter q for the geometric distribution used to select
    /// the insertion layer for newly inserted graph nodes, from the parameter
    /// m_L of the original HNSW paper.
    pub fn layer_probability_from_m_L(m_L: f64) -> f64 {
        (-m_L.recip()).exp()
    }
}

/// An implementation of the HNSW approximate k-nearest neighbors algorithm, based on "Efficient and
/// robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by
/// Malkov and Yashunin, 2020.
///
/// Evaluation and comparison of vector distances are delegated to an implementor of the
/// `VectorStore` trait, and management of the hierarchical graph is delegated to a `GraphMem`
/// struct.
///
/// Graph search in this implementation is optimized to reduce the number of sequential distance
/// evaluation and distance comparison operations, because we use SMPC protocols to implement these
/// basic ops and so the sequential latency introduced by back-and-forth network communication
/// between protocol parties can become significant without batching of operations. See in
/// particular the documentation for the `layer_search_batched` and `layer_search_greedy` functions
/// for details on these search optimizations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HnswSearcher {
    /// Parameters specifying the behavior of HNSW search in different layers.
    pub params: HnswParams,

    /// Operation mode for managing construction and search of graph layers.
    pub layer_mode: LayerMode,

    /// Statistical distribution for layer selection
    pub layer_distribution: LayerDistribution,
}

pub type ConnectPlanV<V> = ConnectPlan<<V as VectorStore>::VectorRef>;

/// Represents the state updates required for insertion of a new node into an HNSW
/// hierarchical graph.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectPlan<Vector: Ord> {
    /// The new vector to insert
    pub inserted_vector: Vector,

    /// List of neighborhood updates to apply
    pub updates: BTreeMap<(Vector, usize), Vec<Vector>>,

    // TODO change to "entrypoints_update", and type `Option<EntryPointsUpdate>`
    /// Whether this update sets the entry point of the HNSW graph to the inserted vector
    pub update_ep: UpdateEntryPoint,
}

/// Represents a graph update of a single node's neighborhood in a graph, given
/// by a tuple `(update_layer, update_vector, new_neighborhood)`.
pub type NbhdUpdate<Vector> = (usize, Vector, Vec<Vector>);

/// Build the updates in the specified layer representing "`inserted_vector`
/// is connected to `neighbors`", and "each item of `neighbors` is connected
/// to links in the same index of `nb_links`"
pub fn build_layer_updates<V: Clone + Ord>(
    inserted_vector: V,
    neighbors: Vec<V>,
    nb_links: Vec<Vec<V>>,
    layer: usize,
) -> BTreeMap<(V, usize), Vec<V>> {
    once(((inserted_vector, layer), neighbors.clone()))
        .chain(izip!(neighbors, nb_links).map(|(nb, nb_nbs)| ((nb, layer), nb_nbs)))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateEntryPoint {
    /// Do not update entry points based on inserted vector
    False,

    /// Set a new unique entry point
    SetUnique { layer: usize },

    /// Add a new item to the set of entry points
    Append { layer: usize },
}

#[allow(non_snake_case)]
impl HnswSearcher {
    /// Construct an HnswSearcher with specified parameters, constructed to use
    /// a unique graph entry point in the dynamic top graph layer.
    pub fn new_standard(ef_constr: usize, ef_search: usize, M: usize) -> Self {
        Self {
            params: HnswParams::new(ef_constr, ef_search, M),
            layer_mode: LayerMode::Standard,
            layer_distribution: LayerDistribution::new_geometric_from_M(M),
        }
    }

    /// Construct an HnswSearcher with specified parameters, constructed to use
    /// linear scan over a set of entry points at a capped maximum graph layer.
    pub fn new_linear_scan(
        ef_constr: usize,
        ef_search: usize,
        M: usize,
        max_graph_layer: usize,
    ) -> Self {
        Self {
            params: HnswParams::new(ef_constr, ef_search, M),
            layer_mode: LayerMode::LinearScan { max_graph_layer },
            layer_distribution: LayerDistribution::new_geometric_from_M(M),
        }
    }

    /// Construct an HnswSearcher with test parameters suitable for exercising
    /// search functionality.
    ///
    /// This function is provided in lieu of a `Default` implementation because
    /// good parameter selections for HNSW search generally depend on the
    /// underlying data distribution, so there isn't a reasonable "generally
    /// applicable" default value.
    pub fn new_with_test_parameters() -> Self {
        let (ef_constr, ef_search, M) = (64, 32, 32);
        Self::new_standard(ef_constr, ef_search, M)
    }

    /// Choose a random insertion layer from the configured distribution,
    /// producing graph layers which decrease in density by a constant factor
    /// per layer.
    ///
    /// The layer generated by this functionality is suitable for use in the
    /// `insert` and `search_to_insert` function calls directly.
    pub fn gen_layer_rng(&self, rng: &mut impl RngCore) -> Result<usize> {
        self.layer_distribution.gen_layer(rng)
    }

    /// Choose a random insertion layer from the configured distribution,
    /// producing graph layers which decrease in density by a constant factor
    /// per layer.
    ///
    /// Generates the layer value based on the evaluation of a keyed PRF on a
    /// hashable input identifier `value`, for instance a vector id or a request
    /// id.
    ///
    /// The layers generated by this functionality is suitable for use in the
    /// `insert` and `search_to_insert` function calls directly.
    pub fn gen_layer_prf<H: Hash>(&self, prf_key: &[u8; 16], value: &H) -> Result<usize> {
        // produce `value_hash` from `value` using `SipHasher13`, keyed with `prf_key`
        let mut hasher = SipHasher13::new_with_key(prf_key);
        value.hash(&mut hasher);
        let value_hash: u64 = hasher.finish();

        // initialize `AesRng` with seed `prf_key ^ (value_hash || value_hash)`
        let mut rng_seed = *prf_key;
        for (idx, byte) in value_hash.to_le_bytes().into_iter().enumerate() {
            rng_seed[idx] ^= byte;
            rng_seed[idx + 8] ^= byte;
        }
        let mut rng = AesRng::from_seed(rng_seed);

        self.gen_layer_rng(&mut rng)
    }

    /// Return a tuple containing:
    /// - An initial candidate neighborhood in the top layer of the graph
    /// - The number of search layers
    /// - A finalized insertion layer
    /// - Enum designating how to handle entry point update
    #[allow(non_snake_case)]
    #[instrument(level = "trace", target = "searcher::cpu_time", skip_all)]
    async fn search_init<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        query: &V::QueryRef,
        insertion_layer: usize,
    ) -> Result<(SortedNeighborhoodV<V>, usize, usize, UpdateEntryPoint)> {
        match self.layer_mode {
            LayerMode::Standard => {
                let ep = graph.get_first_entry_point().await;
                let (W, layer) = self.init_nbhd_from_ep(store, ep, query).await?;

                // Layers are 0-indexed, so number of graph layers is one greater than the entry point layer
                // if an entry point is present, and otherwise 0.
                let n_layers = layer.map(|l| l + 1).unwrap_or(0);

                // Set new entry point if layer is greater than entry point layer, or no entry point available
                let update_ep = if layer.map(|l| insertion_layer > l).unwrap_or(true) {
                    UpdateEntryPoint::SetUnique {
                        layer: insertion_layer,
                    }
                } else {
                    UpdateEntryPoint::False
                };

                Ok((W, n_layers, insertion_layer, update_ep))
            }
            LayerMode::Bounded { max_graph_layer } => {
                let ep = graph.get_first_entry_point().await;
                let (W, layer) = self.init_nbhd_from_ep(store, ep, query).await?;

                // Layers are 0-indexed, so number of graph layers is one greater than the entry point layer
                // if an entry point is present, and otherwise 0.
                let n_layers = layer.map(|l| l + 1).unwrap_or(0);

                // Truncate insertion layer at max graph layer.
                let bounded_insertion_layer = insertion_layer.min(max_graph_layer);

                // Set new entry point if layer is greater than entry point layer, or no entry point available.
                let update_ep = if layer.map(|l| bounded_insertion_layer > l).unwrap_or(true) {
                    UpdateEntryPoint::SetUnique {
                        layer: bounded_insertion_layer,
                    }
                } else {
                    UpdateEntryPoint::False
                };

                Ok((W, n_layers, bounded_insertion_layer, update_ep))
            }
            LayerMode::LinearScan { max_graph_layer } => {
                // Get all valid entry points
                let (ep_vectors, ep_layers): (Vec<_>, Vec<_>) = graph
                    .entry_points
                    .iter()
                    .cloned()
                    .map(|ep| (ep.point, ep.layer))
                    .unzip();
                let ep_vectors = store.only_valid_vectors(ep_vectors).await;

                // TODO when updating entry points, should check for invalid vectors and remove

                // Verify all entry points are at the max graph layer
                if !ep_layers.into_iter().all(|l| l == max_graph_layer) {
                    bail!("Found entry point in invalid graph layer for linear scan functionality");
                }

                let (W, n_layers) = if ep_vectors.is_empty() {
                    let ep = graph.get_temporary_entry_point();
                    let (W, layer) = self.init_nbhd_from_ep(store, ep, query).await?;

                    // Layers are 0-indexed, so number of graph layers is one greater than the entry point layer
                    // if an entry point is present, and otherwise 0.
                    let n_layers = layer.map(|l| l + 1).unwrap_or(0);

                    (W, n_layers)
                } else {
                    let nearest_point =
                        Self::linear_search_min_distance(store, query, ep_vectors).await?;
                    let mut W = SortedNeighborhood::new();
                    W.edges.push(nearest_point);

                    // Entry points are in layer `max_graph_layer`, so number of layers is one more since layers are 0-indexed.
                    let n_layers = max_graph_layer + 1;

                    (W, n_layers)
                };

                // Truncate insertion layer at max graph layer.
                let bounded_insertion_layer = insertion_layer.min(max_graph_layer);

                // Add query to entry points set if target insertion layer is greater than the max graph layer
                let update_ep = if insertion_layer > max_graph_layer {
                    UpdateEntryPoint::Append {
                        layer: max_graph_layer,
                    }
                } else {
                    UpdateEntryPoint::False
                };

                Ok((W, n_layers, bounded_insertion_layer, update_ep))
            }
        }
    }

    /// For a specified entry point, returns an initialized singleton candidate
    /// neighborhood containing the entry point, and the graph layer of this
    /// neighborhood.
    ///
    /// If `ep` is specified as `None` (no entry point available), then returns
    /// an empty neighborhood and `None` for its layer.
    async fn init_nbhd_from_ep<V: VectorStore>(
        &self,
        store: &mut V,
        ep: Option<(V::VectorRef, usize)>,
        query: &V::QueryRef,
    ) -> Result<(SortedNeighborhoodV<V>, Option<usize>)> {
        if let Some((entry_point, layer)) = ep {
            let distance = store.eval_distance(query, &entry_point).await?;

            let mut W = SortedNeighborhood::new();
            W.insert(store, entry_point, distance).await?;

            Ok((W, Some(layer)))
        } else {
            Ok((SortedNeighborhood::new(), None))
        }
    }

    /// Mutate `W` into the `ef` nearest neighbors of query vector `q` in layer `lc` using a
    /// depth-first graph traversal. One of several concrete implementations is selected depending
    /// on `ef`. Terminates when `W` contains vectors which are the nearest to `q` among all
    /// traversed vertices and their neighbors.
    #[instrument(
        level = "trace",
        target = "searcher::cpu_time",
        fields(event_type = Operation::LayerSearch.id()),
        skip(store, graph, q, W))]
    #[allow(non_snake_case)]
    async fn search_layer<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) -> Result<()> {
        match ef {
            0 => {
                bail!("ef cannot be 0");
            }
            1 => {
                let start = W.get_nearest().ok_or(eyre!("W cannot be empty"))?;
                let nearest = Self::layer_search_greedy(store, graph, q, start, lc).await?;

                W.edges.clear();
                W.edges.push(nearest);
            }
            2..32 => {
                Self::layer_search_std(store, graph, q, W, ef, lc).await?;
            }
            _ => {
                Self::layer_search_batched_v2(store, graph, q, W, ef, lc).await?;
            }
        }
        Ok(())
    }

    /// The standard layer search algorithm for HNSW search, which inspects
    /// neighborhoods of graph nodes one element at a time, comparing each
    /// in sequence against the current farthest element of the candidate
    /// neighborhood W and inserting into W if closer.
    ///
    /// This implementation varies slightly from the original specification of
    /// Malkov and Yashunin in that the candidates queue C is removed, and
    /// instead new candidates are identified from W directly by a linear
    /// scan, keeping track of which nodes have been "opened" and had their
    /// neighbors inspected, which is recorded in an additional `HashSet` of
    /// vector ids.
    #[instrument(level = "debug", skip(store, graph, q, W))]
    async fn layer_search_std<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) -> Result<()> {
        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // The set of visited vectors for which we have inspected their neighborhood
        let mut opened = HashSet::<V::VectorRef>::new();

        // fq: The current furthest distance in W.
        let (_, mut fq) = W
            .get_furthest()
            .ok_or(eyre!("No furthest element found"))?
            .clone();

        // These spans accumulate running time of multiple atomic operations
        let eval_dist_span = trace_span!(target: "searcher::cpu_time", "eval_distance_batch_aggr");
        let less_than_span = trace_span!(target: "searcher::cpu_time", "less_than_aggr");
        let insert_span =
            trace_span!(target: "searcher::cpu_time", "insert_into_sorted_neighborhood_aggr");

        // Continue until all current entries in candidate nearest neighbors list have
        // been opened
        while let Some(c) = W
            .edges
            .iter()
            .map(|(c, _)| c)
            .find(|&c| !opened.contains(c))
        {
            // Open the candidate node and visit its unvisited neighbors, computing
            // distances between the query and neighbors as a batch
            let c_links = HnswSearcher::open_node(store, graph, c, lc, q, &mut visited)
                .instrument(eval_dist_span.clone())
                .await?;
            opened.insert(c.clone());
            debug!(event_type = Operation::OpenNode.id(), ef, lc);

            for (e, eq) in c_links.into_iter() {
                if W.len() == ef {
                    // When W is full, we decide whether to replace the furthest element
                    if store
                        .less_than(&eq, &fq)
                        .instrument(less_than_span.clone())
                        .await?
                    {
                        // Make room for the new better candidate...
                        W.pop_furthest();
                    } else {
                        // ...or ignore the candidate and do not continue on this path.
                        continue;
                    }
                }

                // Track the new candidate as a potential k-nearest
                W.insert(store, e, eq)
                    .instrument(insert_span.clone())
                    .await?;

                // fq stays the furthest distance in W
                (_, fq) = W
                    .get_furthest()
                    .ok_or(eyre!("No furthest element found"))?
                    .clone();
            }
        }

        Ok(())
    }

    /// Run an HNSW layer search with batched operation. The algorithm mutates
    /// the input sorted neighborhood `W` into the `ef` (approximate)
    /// nearest vectors to the query `q` in layer `lc` of the layered graph
    /// `graph`.
    ///
    /// As in the standard HNSW layer search, this algorithm proceeds by
    /// sequentially inspecting the neighbors of previously un-opened
    /// entries of `W` closest to `q`, inserting inspected nodes into `W` if
    /// they are nearer to `q` than the current farthest entry of `W`
    /// (or unconditionally if `W` needs to be filled to size `ef`). The
    /// entries of `W` are stored in sorted order of their distance to `q`,
    /// so new nodes are inserted into `W` using a search/sort procedure.
    ///
    /// Distinct from the standard HNSW algorithm, this batched implementation
    /// maintains queues for both distance comparison operations and sorted
    /// list insertion operations so they can be processed in batches rather
    /// than individually. This has to be handled with some care, as
    /// the efficient traversal of the graph layer depends on aspects of the
    /// ongoing sequential search pattern. It is accomplished in the
    /// following way.
    ///
    /// First, as `W` is initially filled up to size `ef`, the entire neighborhoods
    /// of nodes are inserted into `W` via a batched insertion operation
    /// such as a low-depth sorting network. (This functionality is provided
    /// by `SortedNeighborhood::insert_batch`.) This continues
    /// until `W` has reached size `ef`, so that additional insertions will
    /// result in truncation of farthest elements.
    ///
    /// Once `W` has size `ef`, the second phase of graph traversal starts. In
    /// this phase, when an entry of `W` is processed, its neighbors are put
    /// into a queue for later comparison against the entry of `W` farthest
    /// from `q`. The size of this queue is calibrated such that the number
    /// of entries which eventually are inserted into `W` from this comparison
    /// is around a specific targeted batch insertion size.
    ///
    /// However, during graph traversal, the proportion of inspected nodes which
    /// ultimately are inserted into `W` decreases as the neighborhood `W`
    /// better approximates the actual nearest neighbors of `q`. As such,
    /// the number of elements that should be inspected to identify a
    /// particular number of insertion elements increases. To estimate an
    /// appropriate number of elements to compare as a batch against the
    /// farthest element of `W`, the algorithm requires an ongoing estimate
    /// of the rate of insertions relative to the number of inspected nodes.
    ///
    /// This rate estimate is maintained throughout the traversal as the ratio
    /// of a fixed numerator and a dynamic denominator, which is
    /// periodically multiplied by a scaling factor when the estimated rate
    /// is lower than the measured insertion rate over a batch. This
    /// representation is chosen so that the reciprocal of the rate, which
    /// is the factor the target batch insertion size is multiplied by to
    /// estimate the required queue size, is a fixed-precision value which
    /// avoids the issue of numerator and denominator values growing too large
    /// after repeated algebraic manipulations.
    ///
    /// Items meant for insertion are added to a second queue, the insertion
    /// queue. Whenever this queue reaches a fixed target batch insertion
    /// size, elements from the queue are inserted into `W` with a low-depth
    /// batch insertion operation.
    ///
    /// This process of collecting and enqueuing new nodes to inspect, running
    /// comparisons of these nodes against the worst item in `W` in batches,
    /// updating the rate estimate if needed, and inserting full batches of
    /// filtered graph nodes into `W` at once, continues until all items
    /// currently in `W` at the end of a traversal loop have been opened, and
    /// all neighbors inspected.
    ///
    /// At this point, a clean-up step takes place: since the two queues for
    /// batch comparison and batch insertion are only processed when a full
    /// batch is ready, some items may be left over. To ensure that all
    /// enqueued items are handled, first the batch comparison queue and then
    /// the batch insertion queue is processed, which may result in one or more
    /// new elements being inserted into `W`. If new elements are inserted,
    /// then the graph traversal loop continues as before, opening the newly
    /// inserted items in `W`. If no new entries are inserted in this
    /// clean-up step, then the graph traversal is complete, and the function
    /// returns.
    #[instrument(level = "debug", skip(store, graph, q, W))]
    #[allow(dead_code)]
    async fn layer_search_batched<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) -> Result<()> {
        let metrics_labels = [("layer", lc.to_string())];
        let mut metric_edges = FastHistogram::new(&format!("search_edges_layer{}", lc));

        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // The set of visited vectors for which we have inspected their neighborhood
        let mut opened = HashSet::<V::VectorRef>::new();

        // c: the current candidate to be opened, initialized to first entry of W
        let (mut c, _cq) = W.get_nearest().ok_or(eyre!("W cannot be empty"))?.clone();

        // These spans accumulate running time of multiple atomic operations
        let eval_dist_span = trace_span!(target: "searcher::cpu_time", "eval_distance_batch_aggr");
        let less_than_span = trace_span!(target: "searcher::cpu_time", "less_than_aggr");
        let insert_span =
            trace_span!(target: "searcher::cpu_time", "insert_into_sorted_neighborhood_aggr");

        // TODO optimize batch size selection
        let insertion_batch_size = ef / 6;
        let mut insertion_queue: Vec<(V::VectorRef, V::DistanceRef)> =
            Vec::with_capacity(4 * insertion_batch_size);

        let mut comparison_queue: Vec<(V::VectorRef, V::DistanceRef)> = Vec::new();

        // Numerator and denominator of estimated current insertion rate
        const INS_RATE_NUM: usize = 64;
        let mut ins_rate_denom = INS_RATE_NUM;
        const MAX_INS_RATE_DENOM: usize = INS_RATE_NUM * 1000;

        // When the estimated insertion rate is determined to be too low, the value is
        // updated by a fixed factor, rounding up to respect the fixed-precision
        // representation.
        //
        // TODO optimize insertion rate estimate update size
        const INS_RATE_UPDATE: (usize, usize) = (5, 6);

        // The insertion rate after the next update
        let mut next_ins_rate_denom = (ins_rate_denom * INS_RATE_UPDATE.1) / INS_RATE_UPDATE.0;

        // Main graph traversal loop: continue until all graph nodes in the exploration
        // set W have been opened and none of the neighbors are closer to the
        // query than the nodes in W
        let mut depth = 0;
        loop {
            depth += 1;
            // Open the candidate node and visit its unvisited neighbors, computing
            // distances between the query and neighbors as a batch
            let mut c_links = HnswSearcher::open_node(store, graph, &c, lc, q, &mut visited)
                .instrument(eval_dist_span.clone())
                .await?;
            opened.insert(c.clone());
            debug!(event_type = Operation::OpenNode.id(), ef, lc);
            metric_edges.record(c_links.len() as f64);

            // If W is not filled to size ef, insert neighbors in batches until it is
            if W.len() < ef && !c_links.is_empty() {
                let n_insert = c_links.len().min(ef - W.len());
                let batch: Vec<_> = c_links.drain(0..n_insert).collect();
                W.insert_batch(store, &batch)
                    .instrument(insert_span.clone())
                    .await?;
            }

            // Once W is filled, main processing logic for opening nodes happens here
            while !c_links.is_empty() {
                // Add pending comparisons for initial filter pass to a queue. Note that it's
                // the _next_ insertion rate estimate that is used to compute this target batch
                // size so that with reasonably high probability enough insertions are
                // identified to trigger a batch insertion operation.
                let target_batch_size = insertion_batch_size * next_ins_rate_denom / INS_RATE_NUM;
                let n_inspect = c_links
                    .len()
                    .min(target_batch_size.saturating_sub(comparison_queue.len()));
                comparison_queue.extend(c_links.drain(0..n_inspect));

                // Process pending comparisons queue if there are enough elements
                if comparison_queue.len() >= target_batch_size {
                    // Compare elements against current farthest element of W
                    let fq = W
                        .get_furthest()
                        .ok_or(eyre!("No furthest element found"))?
                        .1
                        .clone();
                    let batch: Vec<_> = comparison_queue
                        .iter()
                        .map(|(_c, cq)| (cq.clone(), fq.clone()))
                        .collect();
                    let batch_size = batch.len();

                    let results = store
                        .less_than_batch(&batch)
                        .instrument(less_than_span.clone())
                        .await?;
                    let n_insertions: usize = results.iter().map(|r| *r as usize).sum();
                    debug!(
                        batch_size,
                        n_insertions, "Batch distances comparison filter"
                    );

                    // Enqueue strictly closer elements for insertion
                    let new_insertions = results
                        .into_iter()
                        .zip(comparison_queue.drain(..))
                        .filter_map(|(res, link)| if res { Some(link) } else { None });
                    insertion_queue.extend(new_insertions);

                    // If measured insertion rate is too low, update the estimated insertion rate.
                    //
                    // Update rule: if measured insertion rate is smaller than the harmonic mean of
                    // the current estimated insertion rate and the next update to this rate, then
                    // apply the update.
                    if (ins_rate_denom + next_ins_rate_denom) * n_insertions
                        < 2 * INS_RATE_NUM * batch_size
                        && next_ins_rate_denom < MAX_INS_RATE_DENOM
                    {
                        debug!(
                            prev = ins_rate_denom,
                            new = next_ins_rate_denom,
                            "Update insertion rate estimate denominator"
                        );
                        ins_rate_denom = next_ins_rate_denom;
                        next_ins_rate_denom =
                            (ins_rate_denom * INS_RATE_UPDATE.1) / INS_RATE_UPDATE.0;
                    }
                }

                // Process pending insertions queue if there are enough elements
                while insertion_queue.len() >= insertion_batch_size {
                    let batch: Vec<_> = insertion_queue.drain(0..insertion_batch_size).collect();
                    W.insert_batch(store, &batch)
                        .instrument(insert_span.clone())
                        .await?;
                    W.trim_to_k_nearest(ef);
                }
            }

            // Select next unopened nearest neighbor candidate to open
            let mut c_next = W
                .iter()
                .map(|(c, _)| c)
                .find(|&c| !opened.contains(c))
                .cloned();

            // Once we've opened all elements of candidate neighborhood W, process any
            // remaining elements in the comparison and insertion queues. If
            // new nodes are added to W that need to be opened and processed,
            // continue the graph traversal. Otherwise, halt.

            if c_next.is_none() {
                debug!("Batched layer search clean-up");

                // Step 1: compare neighbors in pending comparisons queue
                if !comparison_queue.is_empty() {
                    // Compare elements against current farthest element of W
                    let fq = W
                        .get_furthest()
                        .ok_or(eyre!("No furthest element found"))?
                        .1
                        .clone();
                    let batch: Vec<_> = comparison_queue
                        .iter()
                        .map(|(_c, cq)| (cq.clone(), fq.clone()))
                        .collect();
                    let batch_size = batch.len();

                    let results = store
                        .less_than_batch(&batch)
                        .instrument(less_than_span.clone())
                        .await?;
                    let n_insertions: usize = results.iter().map(|r| *r as usize).sum();

                    debug!(
                        batch_size,
                        n_insertions, "Batch distances comparison filter"
                    );

                    // Enqueue strictly closer elements for insertion
                    let new_insertions = results
                        .into_iter()
                        .zip(comparison_queue.drain(..))
                        .filter_map(|(res, link)| if res { Some(link) } else { None });
                    insertion_queue.extend(new_insertions);
                }

                // Step 2: insert new neighbors which have passed the filtering step
                for chunk in insertion_queue.chunks(insertion_batch_size) {
                    W.insert_batch(store, chunk)
                        .instrument(insert_span.clone())
                        .await?;
                    W.trim_to_k_nearest(ef);
                }
                insertion_queue.clear();

                // Step 3: try again to select an unopened candidate
                c_next = W
                    .iter()
                    .map(|(c, _)| c)
                    .find(|&c| !opened.contains(c))
                    .cloned();
            }

            match c_next {
                Some(val) => {
                    c = val;
                }
                None => {
                    // All candidates in W have been opened, and all elements pending comparison
                    // and insertion have been inserted, so layer search is complete.
                    break;
                }
            }
        }
        metrics::histogram!("search_depth", &metrics_labels).record(depth as f64);
        Ok(())
    }

    /// Run an HNSW layer search with batched operation. The algorithm mutates
    /// the input sorted neighborhood `W` into the `ef` (approximate) nearest
    /// vectors to the query `q` in layer `lc` of the layered graph `graph`.
    ///
    /// As in the standard HNSW layer search, this algorithm proceeds by
    /// sequentially inspecting the neighbors of previously un-opened entries of
    /// `W` closest to `q`, inserting inspected nodes into `W` if they are
    /// nearer to `q` than the current farthest entry of `W` (or unconditionally
    /// if `W` needs to be filled to size `ef`). The entries of `W` are stored
    /// in sorted order of their distance to `q`, so new nodes are inserted into
    /// `W` using a search/sort procedure.
    ///
    /// Distinct from the standard HNSW algorithm, this batched implementation
    /// opens and processes multiple candidate nodes in batches rather than
    /// individually so that basic operations can be executed in parallel. This
    /// has to be handled with some care, as efficient traversal of the graph
    /// layer depends on aspects of the ongoing sequential search pattern. It is
    /// accomplished in the following way.
    ///
    /// First, as `W` is initially filled up to size `ef`, an initial
    /// breadth-first graph traversal is executed to choose a starting search
    /// neighborhood from which the main search can proceed.  Once neighbors are
    /// chosen, the distances between all new neighbors and the search query are
    /// evaluated as a batch, and the nodes are organized into a candidate
    /// neighborhood by a single sorting operation.
    ///
    /// Once `W` has size `ef`, the second phase of graph traversal starts. In
    /// this phase, batches of unopened elements of `W` are opened and
    /// processed, so that each batch produces approximately a constant number
    /// of "improved" neighbors which are to be included in `W`. During graph
    /// traversal, the proportion of inspected nodes which ultimately are
    /// inserted into `W` decreases as the neighborhood `W` better approximates
    /// the actual nearest neighbors of `q`. To achieve an approximately
    /// constant number of newly inserted elements per batch, an estimate of the
    /// approximate "current insertion rate per inspected node" is maintained,
    /// and this value is used to choose the number of nodes opened in a batch
    /// at each step.
    ///
    /// The insertion rate estimate is maintained throughout the traversal as
    /// the ratio of a fixed numerator and a dynamic denominator, which is
    /// periodically multiplied by a scaling factor when the estimated rate is
    /// lower than the measured insertion rate over a batch. This representation
    /// is chosen so that the reciprocal of the rate, which is the factor the
    /// target batch insertion size is multiplied by to estimate the required
    /// number of elements to inspect, is a fixed-precision value which avoids
    /// the issue of numerator and denominator values growing too large after
    /// repeated algebraic manipulations.
    ///
    /// The process of opening batches of nodes from `W`, filtering to remove
    /// nodes which are not nearer to the query than the current worst item in
    /// `W`, and adding the resulting filtered elements into `W`, continues
    /// until all items currently in `W` at the end of a traversal loop have
    /// been opened, at which point the elements in `W` represent the
    /// approximate nearest neighbors to the query and the function returns.
    ///
    /// Note: the difference between this batched implementation and that in
    /// `layer_search_batched` is that the latter does not batch node openings,
    /// and instead maintains several explicit queues for batched filtering and
    /// batched insertion into `W`.  This achieves a similar effect, but does
    /// not allow batching of distance evaluation, which has become a
    /// significant latency bottleneck in practice.
    #[instrument(level = "debug", skip(store, graph, q, W))]
    async fn layer_search_batched_v2<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) -> Result<()> {
        // These spans accumulate running time of multiple atomic operations
        let eval_dist_span = trace_span!(target: "searcher::cpu_time", "eval_distance_batch_aggr");
        let less_than_span = trace_span!(target: "searcher::cpu_time", "less_than_aggr");
        let insert_span =
            trace_span!(target: "searcher::cpu_time", "insert_into_sorted_neighborhood_aggr");

        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // The set of visited vectors for which we have inspected their neighborhood
        let mut opened = HashSet::new();

        // Fill initial candidate neighborhood from W.  List is constructed as:
        //
        // [w1, w2, ..., wk, ---N(w1)---, ---N(w2)---, ...],
        //
        // where neighborhoods of elements in the list are added to the end,
        // starting with the elements of W, then with the neighbors of w1, then
        // those of w2, and so on, until the list has length at least ef.  This
        // represents a breadth-first traversal of the graph, starting with W.
        // This is done initially without opening nodes so that all distances
        // can be computed in a single batch.
        let mut init_nodes = Vec::from_iter(W.iter().map(|(e, _eq)| e.clone()));
        let mut open_idx = 0;
        while open_idx < init_nodes.len() && init_nodes.len() < ef {
            // get valid, unvisited neighbors of current node at `open_idx`
            let mut nbhd = graph.get_links(&init_nodes[open_idx], lc).await;
            nbhd.retain(|x| !init_nodes.contains(x));
            nbhd = store.only_valid_vectors(nbhd).await;

            // extend `init_nodes` with these neighbors, and progress
            init_nodes.extend(nbhd);
            open_idx += 1;
        }
        // open nodes identified above and insert neighbors into `W`
        let (init_opened, init_links) = HnswSearcher::open_nodes_batch(
            store,
            graph,
            &init_nodes[..open_idx],
            lc,
            q,
            &mut visited,
            None,
        )
        .instrument(eval_dist_span.clone())
        .await?;

        opened.extend(init_opened);

        W.insert_batch(store, &init_links)
            .instrument(insert_span.clone())
            .await?;
        W.trim_to_k_nearest(ef);

        // Target number of elements to insert into candidate neighborhood as a batch.
        //
        // TODO optimize batch size selection
        let insertion_batch_size = ef / 2;

        // Numerator and denominator of estimated current insertion rate
        const INS_RATE_NUM: usize = 64;
        let mut ins_rate_denom = INS_RATE_NUM;
        const MAX_INS_RATE_DENOM: usize = INS_RATE_NUM * 1000;

        // When the estimated insertion rate is determined to be too low, the value is
        // updated by a fixed factor, rounding up to respect the fixed-precision
        // representation.
        //
        // TODO optimize insertion rate estimate update size
        const INS_RATE_UPDATE: (usize, usize) = (5, 6);

        // The insertion rate after the next update
        let mut next_ins_rate_denom = (ins_rate_denom * INS_RATE_UPDATE.1) / INS_RATE_UPDATE.0;

        // Sequential depth of traversal process (number of batch openings) for metrics
        let mut depth = 0;

        // The current list of elements of `W` which are unopened
        let mut cur_unopened = Vec::from_iter(
            W.iter()
                .map(|(c, _)| c)
                .filter(|&c| !opened.contains(c))
                .cloned(),
        );

        // Main graph traversal loop: continue until all graph nodes in the exploration
        // set W have been opened and none of the neighbors are closer to the
        // query than the nodes in W
        while !cur_unopened.is_empty() {
            depth += 1;

            // Estimate the number of neighbors to visit which will result in approximately
            // the desired number of new elements to be inserted into the candidate neighborhood.
            let target_batch_size = insertion_batch_size * ins_rate_denom / INS_RATE_NUM;

            // Open several candidate nodes, visit unvisited neighbors, and compute distances
            // between the query and neighbors as a batch. Opens nodes until at least
            // `target_batch_size` neighbors are visited or all nodes are opened.
            let (new_opened, c_links) = HnswSearcher::open_nodes_batch(
                store,
                graph,
                &cur_unopened,
                lc,
                q,
                &mut visited,
                Some(target_batch_size),
            )
            .instrument(eval_dist_span.clone())
            .await?;

            debug!(
                event_type = Operation::OpenNode.id(),
                increment_amount = new_opened.len(),
                ef,
                lc
            );
            opened.extend(new_opened);

            // Compare elements against current farthest element of W
            let fq = W
                .get_furthest()
                .ok_or(eyre!("No furthest element found"))?
                .1
                .clone();
            let batch: Vec<_> = c_links
                .iter()
                .map(|(_c, cq)| (cq.clone(), fq.clone()))
                .collect();
            let batch_size = batch.len();
            let results = store
                .less_than_batch(&batch)
                .instrument(less_than_span.clone())
                .await?;

            // Filter out elements which are not strictly closer than the current worst candidate
            let filtered_links: Vec<_> = results
                .into_iter()
                .zip(c_links)
                .filter_map(|(res, link)| if res { Some(link) } else { None })
                .collect();

            let n_insertions = filtered_links.len();
            debug!(
                batch_size,
                n_insertions, "Batch distances comparison filter"
            );

            // Insert elements which remain into candidate neighborhood, truncating to length `ef`
            W.insert_batch(store, &filtered_links)
                .instrument(insert_span.clone())
                .await?;
            W.trim_to_k_nearest(ef);

            // If measured insertion rate is too low, update the estimated insertion rate.
            //
            // Update rule: if measured insertion rate is smaller than the harmonic mean of
            // the current estimated insertion rate and the next update to this rate, then
            // apply the update.
            if (ins_rate_denom + next_ins_rate_denom) * n_insertions < 2 * INS_RATE_NUM * batch_size
                && next_ins_rate_denom < MAX_INS_RATE_DENOM
            {
                debug!(
                    prev = ins_rate_denom,
                    new = next_ins_rate_denom,
                    "Update insertion rate estimate denominator"
                );
                ins_rate_denom = next_ins_rate_denom;
                next_ins_rate_denom = (ins_rate_denom * INS_RATE_UPDATE.1) / INS_RATE_UPDATE.0;
            }

            // Refresh the list of currently unopened nodes in the candidate neighborhood `W`
            cur_unopened = Vec::from_iter(
                W.iter()
                    .map(|(c, _)| c)
                    .filter(|&c| !opened.contains(c))
                    .cloned(),
            );
        }

        let metrics_labels = [("layer", lc.to_string())];
        metrics::histogram!("search_depth", &metrics_labels).record(depth as f64);
        Ok(())
    }

    /// Variant of layer search using ef parameter of 1, which conducts a greedy search for the node
    /// with minimum distance from the query.
    #[instrument(level = "debug", skip(store, graph, q, start))]
    async fn layer_search_greedy<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        q: &V::QueryRef,
        start: &(V::VectorRef, V::DistanceRef),
        lc: usize,
    ) -> Result<(V::VectorRef, V::DistanceRef)> {
        // Current node of graph traversal
        let (mut c_vec, mut c_dist) = start.clone();

        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::<V::VectorRef>::new();
        visited.insert(c_vec.clone());

        // These spans accumulate running time of multiple atomic operations
        let eval_dist_span = trace_span!(target: "searcher::cpu_time", "eval_distance_batch_aggr");
        let insert_span =
            trace_span!(target: "searcher::cpu_time", "insert_into_sorted_neighborhood_aggr");

        loop {
            // Open the candidate node and visit its unvisited neighbors, computing
            // distances between the query and neighbors as a batch
            let mut c_links = HnswSearcher::open_node(store, graph, &c_vec, lc, q, &mut visited)
                .instrument(eval_dist_span.clone())
                .await?;
            debug!(event_type = Operation::OpenNode.id(), ef = 1u64, lc);

            // Find minimum distance node also including current node
            c_links.push((c_vec.clone(), c_dist.clone()));

            let network = tree_min(c_links.len());
            apply_swap_network(store, &mut c_links, &network)
                .instrument(insert_span.clone())
                .await?;

            // New closest node from greedy search of neighborhood
            let (n_vec, n_dist) = c_links.first().ok_or(eyre!("No neighbors found"))?.clone();

            // If no neighbors are nearer, return current node; otherwise continue
            if n_vec == c_vec {
                return Ok((c_vec, c_dist));
            } else {
                (c_vec, c_dist) = (n_vec, n_dist);
            }
        }
    }

    /// Linear search over all vectors in the given layer that returns a single nearest neighbor.
    /// This is used in the top layer of the HNSW graph to hide relations between queries and the entry point.
    ///
    /// `vectors` should be pre-filtered for only valid entries.
    async fn linear_search_min_distance<V: VectorStore>(
        store: &mut V,
        query: &V::QueryRef,
        vectors: Vec<V::VectorRef>,
    ) -> Result<(V::VectorRef, V::DistanceRef)> {
        // Compute distances from query to all vectors as a batch
        let distances = store.eval_distance_batch(query, &vectors).await?;
        let distances_with_ids = izip!(vectors, distances).collect_vec();

        // Find the minimum distance and the corresponding vector id
        store.get_argmin_distance(&distances_with_ids).await
    }

    /// Evaluate as a batch the distances between the unvisited neighbors of a given node at a given
    /// graph level with the query, marking unvisited neighbors as visited in the supplied
    /// hashset.
    async fn open_node<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        node: &V::VectorRef,
        lc: usize,
        query: &V::QueryRef,
        visited: &mut HashSet<V::VectorRef>,
    ) -> Result<Vec<(V::VectorRef, V::DistanceRef)>> {
        let neighbors = graph.get_links(node, lc).await;

        let unvisited_neighbors: Vec<_> = neighbors
            .into_iter()
            .filter(|e| visited.insert(e.clone()))
            .collect();

        let valid_neighbors = store.only_valid_vectors(unvisited_neighbors).await;

        let distances = store.eval_distance_batch(query, &valid_neighbors).await?;

        Ok(valid_neighbors
            .into_iter()
            .zip(distances.into_iter())
            .collect())
    }

    /// Evaluate as a batch the distances between the unvisited neighbors of a
    /// list of nodes at a given graph level with the query, marking unvisited
    /// neighbors as visited in the supplied hashset.
    ///
    /// If `limit` is `Some(l)`, then elements of `nodes` will be opened until
    /// *at least* `l` unvisited neighbors have been visited, or until all
    /// elements of `nodes` are opened.  If `limit` is `None`, then all elements
    /// of `nodes` will be opened.
    async fn open_nodes_batch<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        nodes: &[V::VectorRef],
        lc: usize,
        query: &V::QueryRef,
        visited: &mut HashSet<V::VectorRef>,
        limit: Option<usize>,
    ) -> Result<(Vec<V::VectorRef>, Vec<(V::VectorRef, V::DistanceRef)>)> {
        let mut valid_neighbors = Vec::new();
        let mut opened_nodes = Vec::with_capacity(nodes.len());

        for node in nodes {
            let neighbors = graph.get_links(node, lc).await;

            let unvisited_neighbors: Vec<_> = neighbors
                .into_iter()
                .filter(|e| visited.insert(e.clone()))
                .collect();

            valid_neighbors.extend(store.only_valid_vectors(unvisited_neighbors).await);
            opened_nodes.push(node.clone());

            // halt opening once at least limit valid neighbors have been visited, if specified
            if let Some(l) = limit {
                if valid_neighbors.len() >= l {
                    break;
                }
            }
        }

        let distances = store.eval_distance_batch(query, &valid_neighbors).await?;
        let nodes_with_distances = valid_neighbors
            .into_iter()
            .zip(distances.into_iter())
            .collect();

        Ok((opened_nodes, nodes_with_distances))
    }

    /// Search for the `k` nearest neighbors to `query` in `store` using HNSW graph `graph`.
    /// Returns a distance-sorted list (nearest first) of the neighbors.
    ///
    /// Layer search operations use the `ef_search` values of the `HnswParams` struct recorded in
    /// the `params` field of `self`. In the original specification of HNSW this uses a specified
    /// value for `ef` at layer 0 only, and `ef = 1` (greedy search) for all higher layers.
    #[allow(non_snake_case)]
    pub async fn search<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        query: &V::QueryRef,
        k: usize,
    ) -> Result<SortedNeighborhoodV<V>> {
        // insertion layer doesn't matter here because `UpdateEntryPoint` output is ignored
        let (mut W, n_layers, _, _) = self.search_init(store, graph, query, 0).await?;

        // Search from the top layer down to layer 0
        for lc in (0..n_layers).rev() {
            let ef = self.params.get_ef_search(lc);
            Self::search_layer(store, graph, query, &mut W, ef, lc).await?;
        }

        W.trim_to_k_nearest(k);
        Ok(W)
    }

    /// Insert `query` into the HNSW index represented by `store` and `graph`.
    /// Return a `V::VectorRef` representing the inserted vector.
    #[instrument(level = "trace", skip_all, target = "searcher::cpu_time")]
    pub async fn insert<V: VectorStoreMut>(
        &self,
        store: &mut V,
        graph: &mut GraphMem<V::VectorRef>,
        query: &V::QueryRef,
        insertion_layer: usize,
    ) -> Result<V::VectorRef> {
        let (neighbors, update_ep) = self
            .search_to_insert(store, graph, query, insertion_layer)
            .await?;
        let inserted = store.insert(query).await;
        self.insert_from_search_results(store, graph, inserted.clone(), neighbors, update_ep)
            .await?;
        Ok(inserted)
    }

    /// Conduct the search phase of HNSW insertion of `query` into the graph at
    /// a specified insertion layer. Layer search uses the "search" type
    /// `ef_constr` parameter(s) for layers above the insertion layer (1 in
    /// standard HNSW), and the "insertion" type `ef_constr` parameter(s) for
    /// layers at and below the insertion layer (a single fixed `ef_constr`
    /// parameter in standard HNSW).
    ///
    /// The output is a vector of the nearest neighbors found in each insertion
    /// layer, and a boolean indicating if the insertion sets the entry point.
    /// Nearest neighbors are provided in the output for each layer in which the
    /// query is to be inserted, including empty neighbor lists for insertion in
    /// any layers higher than the current entry point.
    ///
    /// If no entry point is initialized for the index, then the insertion will
    /// set `query` as the index entry point.
    ///
    /// Note that the `insertion_layer` input does not have any pre-conditions
    /// on it. The `layer_mode` field of `HnswSearcher` may modify the insertion
    /// layer before operation, e.g. by truncating to a maximum layer height.
    /// This step is handled internally.
    #[instrument(
        level = "trace",
        target = "searcher::cpu_time",
        skip(self, store, graph, query)
    )]
    #[allow(non_snake_case)]
    pub async fn search_to_insert<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V::VectorRef>,
        query: &V::QueryRef,
        insertion_layer: usize,
    ) -> Result<(Vec<SortedNeighborhoodV<V>>, UpdateEntryPoint)> {
        // Initialize candidate neighborhood, index of highest search layer,
        // finalized layer of node insertion, and entry point update outcome.
        let (mut W, n_layers, insertion_layer, update_ep) = self
            .search_init(store, graph, query, insertion_layer)
            .await?;

        // Saved links for insertion layers
        let mut links = Vec::new();

        // Search from the top layer down to layer 0
        for lc in (0..n_layers).rev() {
            let ef = if lc > insertion_layer {
                self.params.get_ef_constr_search(lc)
            } else {
                self.params.get_ef_constr_insert(lc)
            };

            Self::search_layer(store, graph, query, &mut W, ef, lc).await?;

            // Save links in output only for layers in which query is inserted
            if lc <= insertion_layer {
                links.push(W.clone());
            }
        }

        // We inserted top-down, so reverse to match the layer indices (bottom=0)
        links.reverse();

        // If query is to be inserted at a new highest layer as a new entry
        // point, insert additional empty neighborhoods for any new layers
        for _ in links.len()..insertion_layer + 1 {
            links.push(SortedNeighborhood::new());
        }

        assert_eq!(links.len(), insertion_layer + 1);

        Ok((links, update_ep))
    }

    /// Prepare a `ConnectPlan` representing the updates required to insert
    /// `inserted_vector` into `graph` with the specified neighbors `links` and
    /// setting the entry point of the graph if `update_ep` is `true`.  The
    /// `links` vector contains the neighbor lists for the newly inserted node
    /// in different graph layers in which it is to be inserted, starting with
    /// layer 0.  Specified links are inserted as-is, without additional
    /// truncation.
    ///
    /// In this implementation, comparisons required for computing the insertion
    /// indices for updated neighborhoods are done in batches.
    ///
    /// This function call does *not* update `graph`.
    pub async fn insert_prepare<V: VectorStore>(
        &self,
        // this may be used in the future to trim l_links
        store: &mut V,
        graph: &mut GraphMem<V::VectorRef>,
        inserted_vector: V::VectorRef,
        links: Vec<Vec<V::VectorRef>>,
        update_ep: UpdateEntryPoint,
    ) -> Result<ConnectPlanV<V>> {
        let updates = vec![(inserted_vector, links, update_ep)];
        let mut r = self.insert_prepare_batch(store, graph, updates).await?;
        Ok(r.pop()
            .ok_or(eyre!("insert_prepare produced no connect plans"))?)
    }

    /// Prepare connect plans for a batch of graph updates.
    ///
    /// Given a collection of updates, generates a sequence of `ConnectPlan`
    /// structs representing the individual sequential updates from applying the
    /// input updates one after another.  This involves computing the
    /// intermediate state of neighborhoods modified by insertions, and
    /// collecting this intermediate state in corresponding updates.
    ///
    /// After the `ConnectPlan` structs are created, the final state of all
    /// updated neighborhoods is inspected to see if any are too large and need
    /// to be compacted.  All such neighborhoods are compacted in one step at
    /// the end of the function call.
    ///
    /// TODO: finalize batched operation of compaction to minimize latency.
    ///
    /// This function call does *not* update `graph`.
    #[allow(clippy::type_complexity)]
    pub async fn insert_prepare_batch<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &mut GraphMem<V::VectorRef>,
        mut updates: Vec<(V::VectorRef, Vec<Vec<V::VectorRef>>, UpdateEntryPoint)>,
    ) -> Result<Vec<ConnectPlanV<V>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        }

        // Sort all neighborhoods by index
        for (_, links, _) in updates.iter_mut() {
            for l in links.iter_mut() {
                l.sort();
            }
        }

        // Output connect plans
        let mut output_plans: Vec<ConnectPlanV<V>> = Vec::new();
        // Map from vector ids to output connect plan indices
        let mut query_idxs: HashMap<<V as VectorStore>::VectorRef, usize> = HashMap::new();

        // Map `(vector_id, layer) -> Vec<query_id>` recording the query ids
        // which are to be inserted into neighborhoods of nodes, and in what
        // order.  Note input `vector_id` can be a `query_id` if the
        // neighborhood of a subsequent query has been extended to include a
        // previous query in the batch.  `BTreeMap` is used for deterministic
        // iteration order.
        let mut nbhd_updates: BTreeMap<
            (<V as VectorStore>::VectorRef, usize),
            Vec<<V as VectorStore>::VectorRef>,
        > = BTreeMap::new();

        for (idx, (vec, links, update_ep)) in updates.iter().enumerate() {
            // Initialize connect plan for output
            output_plans.push(ConnectPlan {
                inserted_vector: vec.clone(),
                updates: BTreeMap::new(),
                update_ep: update_ep.clone(),
            });
            // Record index of associated vector id
            query_idxs.insert(vec.clone(), idx);

            for (layer, neighbors) in links.iter().enumerate() {
                // Add update for inserting node with outgoing edges in this layer
                output_plans[idx]
                    .updates
                    .insert((vec.clone(), layer), neighbors.clone());

                // Record connections to existing nodes, organized by existing node
                for nb in neighbors.iter() {
                    nbhd_updates
                        .entry((nb.clone(), layer))
                        .or_default()
                        .push(vec.clone());
                }
            }
        }

        // Final updated neighborhood associated with each modified `(vector_id, layer)`
        let mut final_nbhds: BTreeMap<
            (<V as VectorStore>::VectorRef, usize),
            Vec<<V as VectorStore>::VectorRef>,
        > = BTreeMap::new();

        for ((nb, layer), query_ids) in nbhd_updates {
            // Identify the graph neighborhood of `nb` in layer `layer` prior to
            // any updates in the batch
            let mut nb_nbhd = if let Some(idx) = query_idxs.get(&nb) {
                // `nb`` is a query id from the current batch
                let update_entry = updates
                    .get(*idx)
                    .ok_or_eyre("Could not find associated update entry")?;
                let nbhd = update_entry
                    .1
                    .get(layer)
                    .ok_or_eyre("Update entry layer not present")?;
                nbhd.clone()
            } else {
                graph.get_links(&nb, layer).await
            };

            // For each individual update, in order, extend the neighborhood and
            // add as update in the corresopnding connect plan
            for query_id in query_ids {
                // Get the output connect plan associated with `query_id`
                let connect_plan_idx = *query_idxs
                    .get(&query_id)
                    .ok_or_eyre("Could not find associated connect plan index")?;
                let connect_plan = output_plans
                    .get_mut(connect_plan_idx)
                    .ok_or_eyre("Could not find associated connect plan")?;

                // Insert `query_id` into the existing index-sorted neighborhood
                match nb_nbhd.binary_search(&query_id) {
                    Err(i) => nb_nbhd.insert(i, query_id),
                    Ok(_) => tracing::warn!("Attempted to add graph edge which was already present: {nb:?} -> {query_id:?} (layer {layer})"),
                }
                // nb_nbhd.push(query_id);

                // Add update reflecting change to the existing neighborhood
                connect_plan
                    .updates
                    .insert((nb.clone(), layer), nb_nbhd.clone());
            }

            final_nbhds.insert((nb, layer), nb_nbhd);
        }

        // Initial updates without compaction are complete.  Now see if any
        // modified neighborhoods are too large.

        let needs_compaction: BTreeMap<_, _> = final_nbhds
            .into_iter()
            .filter(|((_nb, layer), nb_nbhd)| nb_nbhd.len() > self.params.get_M_limit(*layer))
            .collect();

        // Compute compacted neighborhoods
        // TODO implement batched mode
        let mut compacted: BTreeMap<_, _> = BTreeMap::new();
        for ((id, layer), nbhd) in needs_compaction {
            let max_size = self.params.get_M_max(layer);
            let compacted_nbhd =
                Self::compact_neighborhood(store, id.clone(), &nbhd, max_size).await?;
            compacted.insert((id, layer), compacted_nbhd);
        }

        // Add updates for neighborhood compaction to last connect plan
        let last_plan = output_plans
            .last_mut()
            .ok_or_eyre("Output plans unexpectedly empty")?;
        for ((id, layer), mut compacted_nbhd) in compacted {
            compacted_nbhd.sort();
            last_plan.updates.insert((id, layer), compacted_nbhd);
        }

        Ok(output_plans)
    }

    // TODO switch to batched oblivious min-k and random shuffle
    /// Enforce size constraints on the neighborhood in an oblivious manner
    #[allow(dead_code)]
    async fn compact_neighborhood<V: VectorStore>(
        store: &mut V,
        query: V::VectorRef,
        neighborhood: &[V::VectorRef],
        max_size: usize,
    ) -> Result<Vec<V::VectorRef>> {
        let r = store.vectors_as_queries(vec![query]).await;
        let query = &r[0];
        let link_distances = store.eval_distance_batch(query, neighborhood).await?;
        let sorted_idxs =
            run_quickselect_with_store(&mut (*store), &link_distances, max_size).await?;

        let trimmed_neighborhood = sorted_idxs
            .into_iter()
            .take(max_size)
            .map(|idx| neighborhood[idx].clone())
            .collect();
        Ok(trimmed_neighborhood)
    }

    /// Insert a vector using the search results from `search_to_insert`,
    /// that is the nearest neighbor links at each insertion layer, and a flag
    /// indicating whether the vector is to be inserted as the new entry point.
    #[instrument(
        level = "trace",
        target = "searcher::cpu_time",
        skip(self, store, graph, inserted_vector, links)
    )]
    pub async fn insert_from_search_results<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &mut GraphMem<V::VectorRef>,
        inserted_vector: V::VectorRef,
        links: Vec<SortedNeighborhoodV<V>>,
        update_ep: UpdateEntryPoint,
    ) -> Result<()> {
        // Trim and extract unstructured vector lists
        let mut links_unstructured = Vec::new();
        for (lc, mut l) in links.iter().cloned().enumerate() {
            let m = self.params.get_M(lc);
            l.trim_to_k_nearest(m);
            links_unstructured.push(l.vectors_cloned())
        }

        let plan = self
            .insert_prepare(store, graph, inserted_vector, links_unstructured, update_ep)
            .await?;
        graph.insert_apply(plan).await;
        Ok(())
    }

    pub async fn is_match<V: VectorStore>(
        &self,
        store: &mut V,
        neighbors: &[SortedNeighborhoodV<V>],
    ) -> Result<bool> {
        match neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
        {
            None => Ok(false), // Empty database.
            Some((_, smallest_distance)) => store.is_match(smallest_distance).await,
        }
    }

    pub async fn match_count<V: VectorStore>(
        &self,
        store: &mut V,
        neighbors: &[SortedNeighborhoodV<V>],
    ) -> Result<usize> {
        match neighbors.first() {
            None => Ok(0), // Empty database.
            Some(bottom_layer) => bottom_layer.match_count(store).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::GraphMem};
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::chain;
    use rand::SeedableRng;
    use tokio;

    async fn hnsw_db_helper(db: HnswSearcher, seed: u64) -> Result<()> {
        let vector_store = &mut PlaintextStore::new();
        let graph_store = &mut GraphMem::new();
        let rng = &mut AesRng::seed_from_u64(seed);

        let queries1 = IrisDB::new_random_rng(100, rng)
            .db
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries1.iter() {
            let insertion_layer = db.gen_layer_rng(rng)?;
            let (neighbors, update_ep) = db
                .search_to_insert(vector_store, graph_store, query, insertion_layer)
                .await?;
            assert!(!db.is_match(vector_store, &neighbors).await?);

            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            db.insert_from_search_results(
                vector_store,
                graph_store,
                inserted,
                neighbors,
                update_ep,
            )
            .await?;
        }

        let queries2 = IrisDB::new_random_rng(100, rng)
            .db
            .into_iter()
            .map(Arc::new)
            .collect::<Vec<_>>();

        // Insert the codes with helper function
        for query in queries2.iter() {
            let insertion_layer = db.gen_layer_rng(rng)?;
            db.insert(vector_store, graph_store, query, insertion_layer)
                .await?;
        }

        // Search for the same codes and find matches.
        for query in queries1.iter().chain(queries2.iter()) {
            let neighbors = db.search(vector_store, graph_store, query, 1).await?;
            assert!(db.is_match(vector_store, &[neighbors]).await?);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_hnsw_db_default() -> Result<()> {
        let db = HnswSearcher::new_standard(64, 32, 32);

        hnsw_db_helper(db, 0).await
    }

    #[tokio::test]
    async fn test_hnsw_db_linear_scan() -> Result<()> {
        let db = HnswSearcher::new_linear_scan(64, 32, 32, 1);

        hnsw_db_helper(db, 0).await
    }

    #[tokio::test]
    async fn test_hnsw_db_linear_scan_m() -> Result<()> {
        // Ensure the top layer gets exercised by setting low `M` value
        let db = HnswSearcher::new_linear_scan(64, 32, 4, 1);

        hnsw_db_helper(db, 0).await
    }

    #[tokio::test]
    async fn test_search_to_insert_different_layers() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(42);
        let iris_db = IrisDB::new_random_rng(10, &mut rng);

        let mut queries = iris_db.db.into_iter().map(Arc::new);
        // Duplicate queries for the linear mode tests
        let mut queries_copy = queries.clone();

        // Default mode
        let searcher_default = HnswSearcher::new_with_test_parameters();
        let vector_store_default = &mut PlaintextStore::new();
        let graph_store_default = &mut GraphMem::new();

        for (insertion_layer, expected_nb_len, expected_update_ep) in [
            (0, 1, UpdateEntryPoint::SetUnique { layer: 0 }),
            (0, 1, UpdateEntryPoint::False),
            (1, 2, UpdateEntryPoint::SetUnique { layer: 1 }),
            (1, 2, UpdateEntryPoint::False),
            (2, 3, UpdateEntryPoint::SetUnique { layer: 2 }),
            (2, 3, UpdateEntryPoint::False),
        ] {
            let query = queries.next().unwrap();

            let (neighbors, update_ep) = searcher_default
                .search_to_insert(
                    vector_store_default,
                    graph_store_default,
                    &query,
                    insertion_layer,
                )
                .await?;

            assert_eq!(neighbors.len(), expected_nb_len);
            assert_eq!(update_ep, expected_update_ep);

            searcher_default
                .insert(
                    vector_store_default,
                    graph_store_default,
                    &query,
                    insertion_layer,
                )
                .await?;
        }

        // Linear-scan mode
        let mut searcher_linear = HnswSearcher::new_with_test_parameters();
        searcher_linear.layer_mode = LayerMode::LinearScan { max_graph_layer: 1 };
        let vector_store_linear = &mut PlaintextStore::new();
        let graph_store_linear = &mut GraphMem::new();

        for (insertion_layer, expected_nb_len, expected_update_ep) in [
            (0, 1, UpdateEntryPoint::False),
            (0, 1, UpdateEntryPoint::False),
            (1, 2, UpdateEntryPoint::False),
            (1, 2, UpdateEntryPoint::False),
            (2, 2, UpdateEntryPoint::Append { layer: 1 }),
            (2, 2, UpdateEntryPoint::Append { layer: 1 }),
        ] {
            // Same queries used above
            let query = queries_copy.next().unwrap();

            let (neighbors, update_ep) = searcher_linear
                .search_to_insert(
                    vector_store_linear,
                    graph_store_linear,
                    &query,
                    insertion_layer,
                )
                .await?;

            assert_eq!(neighbors.len(), expected_nb_len);
            assert_eq!(update_ep, expected_update_ep);

            searcher_linear
                .insert(
                    vector_store_linear,
                    graph_store_linear,
                    &query,
                    insertion_layer,
                )
                .await?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_insert_prepare_batch() -> Result<()> {
        // Default mode
        let searcher = HnswSearcher::new_linear_scan(64, 32, 2, 1);
        let vector_store = &mut PlaintextStore::new();
        let graph_store = &mut GraphMem::new();

        let mut ids = vec![];

        let mut rng = AesRng::seed_from_u64(42);
        let iris_db = IrisDB::new_random_rng(10, &mut rng);

        // Insert queries into the vector store
        let queries = iris_db.db.into_iter().map(Arc::new);
        for query in queries {
            let id = vector_store.insert(&query).await;
            ids.push(id);
        }

        // For vectors ids 1 to 5, insert into the graph with each other as neighbors
        for (i, &vector_id) in ids[0..5].iter().enumerate() {
            // Add all other vectors as neighbors (excluding self)
            let nbs = Vec::from_iter(chain!(ids[0..i].iter(), ids[i + 1..5].iter()).cloned());

            // Set entry point for first item
            let update_ep = if i == 0 {
                UpdateEntryPoint::SetUnique { layer: 0 }
            } else {
                UpdateEntryPoint::False
            };

            // Create connect plan for this vector at layer 0
            let connect_plan = ConnectPlan {
                inserted_vector: vector_id,
                updates: BTreeMap::from_iter([((vector_id, 0), nbs)]),
                update_ep,
            };

            // Apply the connect plan to the graph
            graph_store.insert_apply(connect_plan).await;
        }

        // Create an update for inserting vector id 6
        let next_id = ids[5];
        let neighbors = vec![ids[0..5].to_vec()];
        let updates = vec![(next_id, neighbors, UpdateEntryPoint::False)];

        // Prepare the batch insertion
        let connect_plans = searcher
            .insert_prepare_batch(vector_store, graph_store, updates)
            .await?;

        // Verify the connect plan was created correctly
        assert_eq!(connect_plans.len(), 1);
        let plan = &connect_plans[0];
        assert_eq!(plan.update_ep, UpdateEntryPoint::False);
        assert_eq!(plan.updates.len(), 6); // 1 for vector id 6, and 5 others for its neighbors

        // Verify that each neighbor's updated neighborhood has exactly 4
        // elements (the original 4 neighbors plus the newly inserted vector,
        // trimmed to M_max=4 in layer 0)
        for ((id, _lc), nbhd) in &plan.updates {
            if *id != next_id {
                assert_eq!(nbhd.len(), 4);
            }
        }

        Ok(())
    }
}
