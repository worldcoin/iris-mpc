//! Implementation of HNSW algorithm for k-nearest-neighbor search over iris
//! biometric templates with high-latency MPC comparison operations. Based on
//! the `HawkSearcher` class of the `hawk-pack` crate:
//!
//! (<https://github.com/Inversed-Tech/hawk-pack/>)

use super::{
    graph::neighborhood::SortedNeighborhoodV,
    sorting::{binary_search::BinarySearch, swap_network::apply_swap_network, tree_min::tree_min},
    vector_store::VectorStoreMut,
};
use crate::hnsw::{
    graph::neighborhood::SortedEdgeIds, metrics::ops_counter::Operation, GraphMem,
    SortedNeighborhood, VectorStore,
};
use itertools::{izip, Itertools};
use rand::RngCore;
use rand_distr::{Distribution, Geometric};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tracing::{debug, instrument, trace_span, Instrument};

/// The number of explicitly provided parameters for different layers of HNSW
/// search, used by the `HnswParams` struct.
pub const N_PARAM_LAYERS: usize = 5;

/// Struct specifying general parameters for HNSW search.
///
/// Most algorithm and graph properties are specified per-layer by a
/// fixed-length array of size `N_PARAM_LAYERS`, for layers up to
/// `N_PARAM_LAYERS - 1`. Layers larger than this use the last set of
/// parameters.
#[allow(non_snake_case)]
#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    /// The number of neighbors for insertion
    pub M: [usize; N_PARAM_LAYERS],

    /// Maximum number of neighbors allowed per graph node
    pub M_max: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` for search layers during construction
    pub ef_constr_search: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` for insertion layers during construction
    pub ef_constr_insert: [usize; N_PARAM_LAYERS],

    /// Exploration factor `ef` during search
    pub ef_search: [usize; N_PARAM_LAYERS],

    /// Probability `q = 1-p` for geometric distribution of layer densities
    pub layer_probability: f64,
}

#[allow(non_snake_case, clippy::too_many_arguments)]
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
        let ef_constr_search_arr = [1usize; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef_construction; N_PARAM_LAYERS];
        let mut ef_search_arr = [1usize; N_PARAM_LAYERS];
        ef_search_arr[0] = ef_search;
        let layer_probability = (M as f64).recip();

        Self {
            M: M_arr,
            M_max: M_max_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
            layer_probability,
        }
    }

    /// Parameter configuration using fixed exploration factor for all layer
    /// search operations, both for insertion and for search.
    pub fn new_uniform(ef: usize, M: usize) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let ef_constr_search_arr = [ef; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef; N_PARAM_LAYERS];
        let ef_search_arr = [ef; N_PARAM_LAYERS];
        let layer_probability = (M as f64).recip();

        Self {
            M: M_arr,
            M_max: M_max_arr,
            ef_constr_search: ef_constr_search_arr,
            ef_constr_insert: ef_constr_insert_arr,
            ef_search: ef_search_arr,
            layer_probability,
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

    pub fn get_M(&self, lc: usize) -> usize {
        Self::get_val(&self.M, lc)
    }

    pub fn get_M_max(&self, lc: usize) -> usize {
        Self::get_val(&self.M_max, lc)
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

    pub fn get_layer_probability(&self) -> f64 {
        self.layer_probability
    }

    pub fn get_m_L(&self) -> f64 {
        Self::m_L_from_layer_probability(self.layer_probability)
    }

    #[inline(always)]
    /// Select value at index `lc` from the input fixed-size array, or the last
    /// index of this array if `lc` is larger than the array size.
    fn get_val(arr: &[usize; N_PARAM_LAYERS], lc: usize) -> usize {
        arr[lc.min(N_PARAM_LAYERS - 1)]
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
#[derive(Clone, Serialize, Deserialize)]
pub struct HnswSearcher {
    /// Parameters specifying the behavior of HNSW search.
    pub params: HnswParams,
}

pub type ConnectPlanV<V> =
    ConnectPlan<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;
pub type ConnectPlanLayerV<V> =
    ConnectPlanLayer<<V as VectorStore>::VectorRef, <V as VectorStore>::DistanceRef>;

/// Represents the state updates required for insertion of a new node into an HNSW
/// hierarchical graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectPlan<Vector, Distance> {
    /// The new vector to insert
    pub inserted_vector: Vector,

    /// The HNSW graph updates required by insertion. The insertion layer of the new vector
    /// is `layers.len() - 1`.
    pub layers: Vec<ConnectPlanLayer<Vector, Distance>>,

    /// Whether this update sets the entry point of the HNSW graph to the inserted vector
    pub set_ep: bool,
}

/// Represents the state updates required for insertion of a new node into a single layer of
/// an HNSW hierarchical graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectPlanLayer<Vector, Distance> {
    /// The neighbors of the inserted vector, and their distances
    pub neighbors: SortedNeighborhood<Vector, Distance>,

    /// `nb_links[i]` is the updated neighborhood of node `neighbors[i]` after the insertion
    pub nb_links: Vec<SortedEdgeIds<Vector>>,
}

impl Default for HnswSearcher {
    fn default() -> Self {
        HnswSearcher {
            params: HnswParams::new(64, 32, 32),
        }
    }
}

#[allow(non_snake_case)]
impl HnswSearcher {
    /// Choose a random insertion layer from a geometric distribution, producing
    /// graph layers which decrease in density by a constant factor per layer.
    pub fn select_layer(&self, rng: &mut impl RngCore) -> usize {
        let p_geom = 1f64 - self.params.get_layer_probability();
        let geom_distr = Geometric::new(p_geom).unwrap();

        geom_distr.sample(rng) as usize
    }

    /// Return a tuple containing a distance-sorted list initialized with the
    /// entry point for the HNSW graph search (with distance to the query
    /// pre-computed), and the number of search layers of the graph hierarchy,
    /// that is, the layer of the entry point plus 1.
    ///
    /// If no entry point is initialized, returns an empty list and layer 0.
    #[allow(non_snake_case)]
    #[instrument(level = "trace", target = "searcher::cpu_time", skip_all)]
    async fn search_init<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V>,
        query: &V::QueryRef,
    ) -> (SortedNeighborhoodV<V>, usize) {
        if let Some((entry_point, layer)) = graph.get_entry_point().await {
            let distance = store.eval_distance(query, &entry_point).await;

            let mut W = SortedNeighborhood::new();
            W.insert(store, entry_point, distance).await;

            (W, layer + 1)
        } else {
            (SortedNeighborhood::new(), 0)
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
        graph: &GraphMem<V>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) {
        match ef {
            0 => {
                panic!("ef cannot be 0");
            }
            1 => {
                let start = W.get_nearest().expect("W cannot be empty");
                let nearest = Self::layer_search_greedy(store, graph, q, start, lc).await;

                W.edges.clear();
                W.edges.push(nearest);
            }
            2..32 => {
                Self::layer_search_std(store, graph, q, W, ef, lc).await;
            }
            _ => {
                Self::layer_search_batched(store, graph, q, W, ef, lc).await;
            }
        }
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
        graph: &GraphMem<V>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) {
        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // The set of visited vectors for which we have inspected their neighborhood
        let mut opened = HashSet::<V::VectorRef>::new();

        // fq: The current furthest distance in W.
        let (_, mut fq) = W.get_furthest().expect("W cannot be empty").clone();

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
                .await;
            opened.insert(c.clone());
            debug!(event_type = Operation::OpenNode.id(), ef, lc);

            for (e, eq) in c_links.into_iter() {
                if W.len() == ef {
                    // When W is full, we decide whether to replace the furthest element
                    if store
                        .less_than(&eq, &fq)
                        .instrument(less_than_span.clone())
                        .await
                    {
                        // Make room for the new better candidate...
                        W.pop_furthest();
                    } else {
                        // ...or ignore the candidate and do not continue on this path.
                        continue;
                    }
                }

                // Track the new candidate as a potential k-nearest
                W.insert(store, e, eq).instrument(insert_span.clone()).await;

                // fq stays the furthest distance in W
                (_, fq) = W.get_furthest().expect("W cannot be empty").clone();
            }
        }
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
    async fn layer_search_batched<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V>,
        q: &V::QueryRef,
        W: &mut SortedNeighborhoodV<V>,
        ef: usize,
        lc: usize,
    ) {
        // The set of vectors which have been considered as potential neighbors
        let mut visited = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // The set of visited vectors for which we have inspected their neighborhood
        let mut opened = HashSet::<V::VectorRef>::new();

        // c: the current candidate to be opened, initialized to first entry of W
        let (mut c, _cq) = W.get_nearest().expect("W cannot be empty").clone();

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
        loop {
            // Open the candidate node and visit its unvisited neighbors, computing
            // distances between the query and neighbors as a batch
            let mut c_links = HnswSearcher::open_node(store, graph, &c, lc, q, &mut visited)
                .instrument(eval_dist_span.clone())
                .await;
            opened.insert(c.clone());
            debug!(event_type = Operation::OpenNode.id(), ef, lc);

            // If W is not filled to size ef, insert neighbors in batches until it is
            if W.len() < ef && !c_links.is_empty() {
                let n_insert = c_links.len().min(ef - W.len());
                let batch: Vec<_> = c_links.drain(0..n_insert).collect();
                W.insert_batch(store, &batch)
                    .instrument(insert_span.clone())
                    .await;
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
                    let fq = W.get_furthest().unwrap().1.clone();
                    let batch: Vec<_> = comparison_queue
                        .iter()
                        .map(|(_c, cq)| (cq.clone(), fq.clone()))
                        .collect();
                    let batch_size = batch.len();

                    let results = store
                        .less_than_batch(&batch)
                        .instrument(less_than_span.clone())
                        .await;
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
                        .await;
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
                    let fq = W.get_furthest().unwrap().1.clone();
                    let batch: Vec<_> = comparison_queue
                        .iter()
                        .map(|(_c, cq)| (cq.clone(), fq.clone()))
                        .collect();
                    let batch_size = batch.len();

                    let results = store
                        .less_than_batch(&batch)
                        .instrument(less_than_span.clone())
                        .await;
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
                        .await;
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
    }

    /// Variant of layer search using ef parameter of 1, which conducts a greedy search for the node
    /// with minimum distance from the query.
    #[instrument(level = "debug", skip(store, graph, q, start))]
    async fn layer_search_greedy<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V>,
        q: &V::QueryRef,
        start: &(V::VectorRef, V::DistanceRef),
        lc: usize,
    ) -> (V::VectorRef, V::DistanceRef) {
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
                .await;
            debug!(event_type = Operation::OpenNode.id(), ef = 1u64, lc);

            // Find minimum distance node also including current node
            c_links.push((c_vec.clone(), c_dist.clone()));

            let network = tree_min(c_links.len());
            apply_swap_network(store, &mut c_links, &network)
                .instrument(insert_span.clone())
                .await;

            // New closest node from greedy search of neighborhood
            let (n_vec, n_dist) = c_links.first().unwrap().clone();

            // If no neighbors are nearer, return current node; otherwise continue
            if n_vec == c_vec {
                return (c_vec, c_dist);
            } else {
                (c_vec, c_dist) = (n_vec, n_dist);
            }
        }
    }

    /// Evaluate as a batch the distances between the unvisited neighbors of a given node at a given
    /// graph level with the query, marking unvisited neighbors as visited in the supplied
    /// hashset.
    async fn open_node<V: VectorStore>(
        store: &mut V,
        graph: &GraphMem<V>,
        node: &V::VectorRef,
        lc: usize,
        query: &V::QueryRef,
        visited: &mut HashSet<V::VectorRef>,
    ) -> Vec<(V::VectorRef, V::DistanceRef)> {
        let neighbors = graph.get_links(node, lc).await;

        let unvisited_neighbors: Vec<_> = neighbors
            .0
            .into_iter()
            .filter(|e| visited.insert(e.clone()))
            .collect();

        let distances = store
            .eval_distance_batch(&[query.clone()], &unvisited_neighbors)
            .await;

        unvisited_neighbors
            .into_iter()
            .zip(distances.into_iter())
            .collect()
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
        graph: &mut GraphMem<V>,
        query: &V::QueryRef,
        k: usize,
    ) -> SortedNeighborhoodV<V> {
        let (mut W, layer_count) = self.search_init(store, graph, query).await;

        // Search from the top layer down to layer 0
        for lc in (0..layer_count).rev() {
            let ef = self.params.get_ef_search(lc);
            Self::search_layer(store, graph, query, &mut W, ef, lc).await;
        }

        W.trim_to_k_nearest(k);
        W
    }

    /// Insert `query` into the HNSW index represented by `store` and `graph`.
    /// Return a `V::VectorRef` representing the inserted vector.
    #[instrument(level = "trace", skip_all, target = "searcher::cpu_time")]
    pub async fn insert<V: VectorStoreMut>(
        &self,
        store: &mut V,
        graph: &mut GraphMem<V>,
        query: &V::QueryRef,
        rng: &mut impl RngCore,
    ) -> V::VectorRef {
        let insertion_layer = self.select_layer(rng);
        let (neighbors, set_ep) = self
            .search_to_insert(store, graph, query, insertion_layer)
            .await;
        let inserted = store.insert(query).await;
        self.insert_from_search_results(store, graph, inserted.clone(), neighbors, set_ep)
            .await;
        inserted
    }

    /// Conduct the search phase of HNSW insertion of `query` into the graph at
    /// a specified insertion layer. Layer search uses the "search" type
    /// `ef_constr` parameter(s) for layers above the insertion layer (1 in
    /// standard HNSW), and the "insertion" type `ef_constr` parameter(s) for
    /// layers at and below the insertion layer (a single fixed `ef_constr` parameter
    /// in standard HNSW).
    ///
    /// The output is a vector of the nearest neighbors found in each insertion
    /// layer, and a boolean indicating if the insertion sets the entry point.
    /// Nearest neighbors are provided in the output for each layer in which
    /// the query is to be inserted, including empty neighbor lists for
    /// insertion in any layers higher than the current entry point.
    ///
    /// If no entry point is initialized for the index, then the insertion will
    /// set `query` as the index entry point.
    #[instrument(
        level = "trace",
        target = "searcher::cpu_time",
        skip(self, store, graph, query)
    )]
    #[allow(non_snake_case)]
    pub async fn search_to_insert<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V>,
        query: &V::QueryRef,
        insertion_layer: usize,
    ) -> (Vec<SortedNeighborhoodV<V>>, bool) {
        let mut links = vec![];

        let (mut W, n_layers) = self.search_init(store, graph, query).await;

        // Search from the top layer down to layer 0
        for lc in (0..n_layers).rev() {
            let ef = if lc > insertion_layer {
                self.params.get_ef_constr_search(lc)
            } else {
                self.params.get_ef_constr_insert(lc)
            };
            Self::search_layer(store, graph, query, &mut W, ef, lc).await;

            // Save links in output only for layers in which query is inserted
            if lc <= insertion_layer {
                links.push(W.clone());
            }
        }

        // We inserted top-down, so reverse to match the layer indices (bottom=0)
        links.reverse();

        // If query is to be inserted at a new highest layer as a new entry
        // point, insert additional empty neighborhoods for any new layers
        let set_ep = insertion_layer + 1 > n_layers;
        for _ in links.len()..insertion_layer + 1 {
            links.push(SortedNeighborhood::new());
        }
        debug_assert!(links.len() == insertion_layer + 1);

        (links, set_ep)
    }

    /// Prepare a `ConnectPlan` representing the updates required to insert `inserted_vector`
    /// into `graph` with the specified neighbors `links` and setting the entry point of the
    /// graph if `set_ep` is `true`.  The `links` vector contains the neighbor lists for the
    /// newly inserted node in different graph layers in which it is to be inserted, starting
    /// with layer 0.
    ///
    /// In this implementation, comparisons required for computing the insertion indices for
    /// updated neighborhoods are done in batches.
    ///
    /// This function call does *not* update `graph`.
    pub async fn insert_prepare<V: VectorStore>(
        &self,
        store: &mut V,
        graph: &GraphMem<V>,
        inserted_vector: V::VectorRef,
        mut links: Vec<SortedNeighborhoodV<V>>,
        set_ep: bool,
    ) -> ConnectPlanV<V> {
        let mut plan = ConnectPlan {
            inserted_vector: inserted_vector.clone(),
            layers: vec![],
            set_ep,
        };

        // Truncate search results to size M before insertion
        for (lc, l_links) in links.iter_mut().enumerate() {
            let M = self.params.get_M(lc);
            l_links.trim_to_k_nearest(M);
        }

        struct NeighborUpdate<Query, Vector, Distance> {
            /// The distance between the vector being inserted to a base vector.
            nb_dist: Distance,
            /// The base vector that we connect to. It is in "query" form to compare to `nb_links`.
            nb_query: Query,
            /// The neighborhood of the base vector.
            nb_links: SortedEdgeIds<Vector>,
            /// The current state of the search.
            search: BinarySearch,
        }

        // Collect current neighborhoods of new neighbors in each layer and
        // initialize binary search
        let mut neighbors = Vec::new();
        for (lc, l_links) in links.iter().enumerate() {
            let nb_queries = store.vectors_as_queries(l_links.vectors_cloned()).await;

            let mut l_neighbors = Vec::with_capacity(l_links.len());
            for ((nb, nb_dist), nb_query) in izip!(l_links.iter(), nb_queries) {
                let nb_links = graph.get_links(nb, lc).await;
                let search = BinarySearch {
                    left: 0,
                    right: nb_links.len(),
                };
                let neighbor = NeighborUpdate {
                    nb_dist: nb_dist.clone(),
                    nb_query,
                    nb_links,
                    search,
                };
                l_neighbors.push(neighbor);
            }
            neighbors.push(l_neighbors);
        }

        // Run searches until completion, executing comparisons in batches
        let mut searches_ongoing: Vec<_> = neighbors
            .iter_mut()
            .flatten()
            .filter(|n| !n.search.is_finished())
            .collect();

        while !searches_ongoing.is_empty() {
            // Find the next batch of distances to evaluate.
            // This is each base neighbor versus the next search position in its neighborhood.
            let dist_batch = searches_ongoing
                .iter()
                .map(|n| {
                    let cmp_idx = n.search.next().unwrap();
                    (n.nb_query.clone(), n.nb_links[cmp_idx].clone())
                })
                .collect_vec();

            // Compute the distances.
            let link_distances = store.eval_distance_pairs(&dist_batch).await;

            // Prepare a batch of less_than.
            // This is |inserted--base| versus |base--neighborhood|.
            let lt_batch = izip!(&searches_ongoing, link_distances)
                .map(|(n, link_dist)| (n.nb_dist.clone(), link_dist))
                .collect_vec();

            // Compute the less_than.
            let results = store.less_than_batch(&lt_batch).await;

            searches_ongoing
                .iter_mut()
                .zip(results)
                .for_each(|(n, res)| {
                    n.search.update(res);
                });

            searches_ongoing.retain(|n| !n.search.is_finished());
        }

        // Directly insert new vector into neighborhoods from search results
        neighbors
            .iter_mut()
            .enumerate()
            .for_each(|(lc, l_neighbors)| {
                let max_links = self.params.get_M_max(lc);
                l_neighbors.iter_mut().for_each(|n| {
                    let insertion_idx = n.search.result().unwrap();
                    n.nb_links.insert(insertion_idx, inserted_vector.clone());
                    n.nb_links.trim_to_k_nearest(max_links);
                });
            });

        // Generate ConnectPlanLayer structs
        plan.layers = links
            .into_iter()
            .zip(neighbors)
            .map(|(l_links, l_neighbors)| ConnectPlanLayer {
                neighbors: l_links,
                nb_links: l_neighbors.into_iter().map(|n| n.nb_links).collect_vec(),
            })
            .collect();

        plan
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
        graph: &mut GraphMem<V>,
        inserted_vector: V::VectorRef,
        links: Vec<SortedNeighborhoodV<V>>,
        set_ep: bool,
    ) {
        let plan = self
            .insert_prepare(store, graph, inserted_vector, links, set_ep)
            .await;
        graph.insert_apply(plan).await;
    }

    pub async fn is_match<V: VectorStore>(
        &self,
        store: &mut V,
        neighbors: &[SortedNeighborhoodV<V>],
    ) -> bool {
        match neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
        {
            None => false, // Empty database.
            Some((_, smallest_distance)) => store.is_match(smallest_distance).await,
        }
    }

    pub async fn match_count<V: VectorStore>(
        &self,
        store: &mut V,
        neighbors: &[SortedNeighborhoodV<V>],
    ) -> usize {
        match neighbors.first() {
            None => 0, // Empty database.
            Some(bottom_layer) => bottom_layer.match_count(store).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::GraphMem};
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_hnsw_db() {
        let vector_store = &mut PlaintextStore::default();
        let graph_store = &mut GraphMem::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let db = HnswSearcher::default();

        let queries1 = IrisDB::new_random_rng(100, rng)
            .db
            .into_iter()
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes.
        for query in queries1.iter() {
            let insertion_layer = db.select_layer(rng);
            let (neighbors, set_ep) = db
                .search_to_insert(vector_store, graph_store, query, insertion_layer)
                .await;
            assert!(!db.is_match(vector_store, &neighbors).await);
            // Insert the new vector into the store.
            let inserted = vector_store.insert(query).await;
            db.insert_from_search_results(vector_store, graph_store, inserted, neighbors, set_ep)
                .await;
        }

        let queries2 = IrisDB::new_random_rng(100, rng)
            .db
            .into_iter()
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes with helper function
        for query in queries2.iter() {
            db.insert(vector_store, graph_store, query, rng).await;
        }

        // Search for the same codes and find matches.
        for query in queries1.iter().chain(queries2.iter()) {
            let neighbors = db.search(vector_store, graph_store, query, 1).await;
            println!("query: {query:?}");
            println!("neighbors: {neighbors:?}");
            assert!(db.is_match(vector_store, &[neighbors]).await);
        }
    }
}
