//* Implementation of HNSW algorithm for k-nearest-neighbor search over iris
//* biometric templates with high-latency MPC comparison operations.  Based on
//* the `HawkSearcher` class of the hawk-pack crate:
//*
//* https://github.com/Inversed-Tech/hawk-pack/

pub use hawk_pack::data_structures::queue::{
    FurthestQueue, FurthestQueueV, NearestQueue, NearestQueueV,
};
use hawk_pack::{GraphStore, VectorStore};
use rand::RngCore;
use rand_distr::{Distribution, Geometric};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// Specify construction and search parameters by layer up to this value minus 1
// any higher layers will use the last set of parameters
pub const N_PARAM_LAYERS: usize = 5;

#[allow(non_snake_case)]
#[derive(PartialEq, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    pub M:                 [usize; N_PARAM_LAYERS], // number of neighbors for insertion
    pub M_max:             [usize; N_PARAM_LAYERS], // maximum number of neighbors
    pub ef_constr_search:  [usize; N_PARAM_LAYERS], // ef_constr for search layers
    pub ef_constr_insert:  [usize; N_PARAM_LAYERS], // ef_constr for insertion layers
    pub ef_search:         [usize; N_PARAM_LAYERS], // ef for search
    pub layer_probability: f64,                     /* p for geometric distribution of layer
                                                     * densities */
}

#[allow(non_snake_case, clippy::too_many_arguments)]
impl HnswParams {
    /// Construct a `Params` object corresponding to parameter configuration
    /// providing the functionality described in the original HNSW paper:
    /// - ef_construction exploration factor used for insertion layers
    /// - ef_search exploration factor used for layer 0 in search
    /// - higher layers in both insertion and search use exploration factor 1,
    ///   representing simple greedy search
    /// - vertex degrees bounded by M_max = M in positive layer graphs
    /// - vertex degrees bounded by M_max0 = 2*M in layer 0 graph
    /// - m_L = 1 / ln(M) so that layer density decreases by a factor of M at
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

    /// Same as standard constructor but with an extra input for a non-standard
    /// `layer_probability` parameter.
    pub fn new_with_layer_probability(
        ef_construction: usize,
        ef_search: usize,
        M: usize,
        layer_probability: f64,
    ) -> Self {
        let M_arr = [M; N_PARAM_LAYERS];
        let mut M_max_arr = [M; N_PARAM_LAYERS];
        M_max_arr[0] = 2 * M;
        let ef_constr_search_arr = [1usize; N_PARAM_LAYERS];
        let ef_constr_insert_arr = [ef_construction; N_PARAM_LAYERS];
        let mut ef_search_arr = [1usize; N_PARAM_LAYERS];
        ef_search_arr[0] = ef_search;

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

/// An implementation of the HNSW algorithm.
///
/// Operations on vectors are delegated to a VectorStore.
/// Operations on the graph are delegated to a GraphStore.
#[derive(Clone, Serialize, Deserialize)]
pub struct HnswSearcher {
    pub params: HnswParams,
}

// TODO remove default value; this varies too much between applications
// to make sense to specify something "obvious"
impl Default for HnswSearcher {
    fn default() -> Self {
        HnswSearcher {
            params: HnswParams::new(64, 32, 32),
        }
    }
}

#[allow(non_snake_case)]
impl HnswSearcher {
    async fn connect_bidir<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        q: &V::VectorRef,
        mut neighbors: FurthestQueueV<V>,
        lc: usize,
    ) {
        let M = self.params.get_M(lc);
        let max_links = self.params.get_M_max(lc);

        neighbors.trim_to_k_nearest(M);

        // Connect all n -> q.
        for (n, nq) in neighbors.iter() {
            let mut links = graph_store.get_links(n, lc).await;
            links.insert(vector_store, q.clone(), nq.clone()).await;
            links.trim_to_k_nearest(max_links);
            graph_store.set_links(n.clone(), links, lc).await;
        }

        // Connect q -> all n.
        graph_store.set_links(q.clone(), neighbors, lc).await;
    }

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
    async fn search_init<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
    ) -> (FurthestQueueV<V>, usize) {
        if let Some((entry_point, layer)) = graph_store.get_entry_point().await {
            let distance = vector_store.eval_distance(query, &entry_point).await;

            let mut W = FurthestQueueV::<V>::new();
            W.insert(vector_store, entry_point, distance).await;

            (W, layer + 1)
        } else {
            (FurthestQueue::new(), 0)
        }
    }

    /// Mutate `W` into the `ef` nearest neighbors of query vector `q` in the
    /// given layer using depth-first graph traversal,  Terminates when `W`
    /// contains vectors which are the nearest to `q` among all traversed
    /// vertices and their neighbors.
    #[allow(non_snake_case)]
    async fn search_layer<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        q: &V::QueryRef,
        W: &mut FurthestQueueV<V>,
        ef: usize,
        lc: usize,
    ) {
        // v: The set of already visited vectors.
        let mut v = HashSet::<V::VectorRef>::from_iter(W.iter().map(|(e, _eq)| e.clone()));

        // C: The set of vectors to visit, ordered by increasing distance to the query.
        let mut C = NearestQueue::from_furthest_queue(W);

        // fq: The current furthest distance in W.
        let (_, mut fq) = W.get_furthest().expect("W cannot be empty").clone();

        while C.len() > 0 {
            let (c, cq) = C.pop_nearest().expect("C cannot be empty").clone();

            // If the nearest distance to C is greater than the furthest distance in W, then
            // we can stop.
            if vector_store.less_than(&fq, &cq).await {
                break;
            }

            // Open the node c and explore its neighbors.

            // Visit all neighbors of c.
            let c_links = graph_store.get_links(&c, lc).await;

            // Evaluate the distances of the neighbors to the query, as a batch.
            let c_links = {
                let e_batch = c_links
                    .iter()
                    .map(|(e, _ec)| e.clone())
                    .filter(|e| {
                        // Visit any node at most once.
                        v.insert(e.clone())
                    })
                    .collect::<Vec<_>>();

                let distances = vector_store.eval_distance_batch(q, &e_batch).await;

                e_batch
                    .into_iter()
                    .zip(distances.into_iter())
                    .collect::<Vec<_>>()
            };

            for (e, eq) in c_links.into_iter() {
                if W.len() == ef {
                    // When W is full, we decide whether to replace the furthest element.
                    if vector_store.less_than(&eq, &fq).await {
                        // Make room for the new better candidate…
                        W.pop_furthest();
                    } else {
                        // …or ignore the candidate and do not continue on this path.
                        continue;
                    }
                }

                // Track the new candidate in C so we will continue this path later.
                C.insert(vector_store, e.clone(), eq.clone()).await;

                // Track the new candidate as a potential k-nearest.
                W.insert(vector_store, e, eq).await;

                // fq stays the furthest distance in W.
                (_, fq) = W.get_furthest().expect("W cannot be empty").clone();
            }
        }
    }

    #[allow(non_snake_case)]
    pub async fn search<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
        k: usize,
    ) -> FurthestQueueV<V> {
        let (mut W, layer_count) = self.search_init(vector_store, graph_store, query).await;

        // Search from the top layer down to layer 0
        for lc in (0..layer_count).rev() {
            let ef = self.params.get_ef_search(lc);
            self.search_layer(vector_store, graph_store, query, &mut W, ef, lc)
                .await;
        }

        W.trim_to_k_nearest(k);
        W
    }

    /// Insert `query` into HNSW index represented by `vector_store` and
    /// `graph_store`.  Return a `V::VectorRef` representing the inserted
    /// vector.
    pub async fn insert<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
        rng: &mut impl RngCore,
    ) -> V::VectorRef {
        let insertion_layer = self.select_layer(rng);
        let (neighbors, set_ep) = self
            .search_to_insert(vector_store, graph_store, query, insertion_layer)
            .await;
        let inserted = vector_store.insert(query).await;
        self.insert_from_search_results(
            vector_store,
            graph_store,
            inserted.clone(),
            neighbors,
            set_ep,
        )
        .await;
        inserted
    }

    /// Conduct the search phase of HNSW insertion of `query` into the graph at
    /// a specified insertion layer.  Layer search uses the "search" type
    /// `ef_constr` parameter(s) for layers above the insertion layer (1 in
    /// standard HNSW), and the "insertion" type `ef_constr` parameter(s) for
    /// layers below the insertion layer (a single fixed `ef_constr` parameter
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
    #[allow(non_snake_case)]
    pub async fn search_to_insert<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        query: &V::QueryRef,
        insertion_layer: usize,
    ) -> (Vec<FurthestQueueV<V>>, bool) {
        let mut links = vec![];

        let (mut W, n_layers) = self.search_init(vector_store, graph_store, query).await;

        // Search from the top layer down to layer 0
        for lc in (0..n_layers).rev() {
            let ef = if lc > insertion_layer {
                self.params.get_ef_constr_search(lc)
            } else {
                self.params.get_ef_constr_insert(lc)
            };
            self.search_layer(vector_store, graph_store, query, &mut W, ef, lc)
                .await;

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
            links.push(FurthestQueue::new());
        }
        debug_assert!(links.len() == insertion_layer + 1);

        (links, set_ep)
    }

    /// Insert a vector using the search results from `search_to_insert`,
    /// that is the nearest neighbor links at each insertion layer, and a flag
    /// indicating whether the vector is to be inserted as the new entry point.
    pub async fn insert_from_search_results<V: VectorStore, G: GraphStore<V>>(
        &self,
        vector_store: &mut V,
        graph_store: &mut G,
        inserted_vector: V::VectorRef,
        links: Vec<FurthestQueueV<V>>,
        set_ep: bool,
    ) {
        // If required, set vector as new entry point
        if set_ep {
            let insertion_layer = links.len() - 1;
            graph_store
                .set_entry_point(inserted_vector.clone(), insertion_layer)
                .await;
        }

        // Connect the new vector to its neighbors in each layer.
        for (lc, layer_links) in links.into_iter().enumerate().rev() {
            self.connect_bidir(vector_store, graph_store, &inserted_vector, layer_links, lc)
                .await;
        }
    }

    pub async fn is_match<V: VectorStore>(
        &self,
        vector_store: &mut V,
        neighbors: &[FurthestQueueV<V>],
    ) -> bool {
        match neighbors
            .first()
            .and_then(|bottom_layer| bottom_layer.get_nearest())
        {
            None => false, // Empty database.
            Some((_, smallest_distance)) => vector_store.is_match(smallest_distance).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_prng::AesRng;
    use hawk_pack::{
        graph_store::graph_mem::GraphMem, vector_store::lazy_memory_store::LazyMemoryStore,
    };
    use rand::SeedableRng;
    use tokio;

    #[tokio::test]
    async fn test_hnsw_db() {
        let vector_store = &mut LazyMemoryStore::new();
        let graph_store = &mut GraphMem::new();
        let rng = &mut AesRng::seed_from_u64(0_u64);
        let db = HnswSearcher::default();

        let queries1 = (0..100)
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

        let queries2 = (101..200)
            .map(|raw_query| vector_store.prepare_query(raw_query))
            .collect::<Vec<_>>();

        // Insert the codes with helper function
        for query in queries2.iter() {
            db.insert(vector_store, graph_store, query, rng).await;
        }

        // Search for the same codes and find matches.
        for query in queries1.iter().chain(queries2.iter()) {
            let neighbors = db.search(vector_store, graph_store, query, 1).await;
            assert!(db.is_match(vector_store, &[neighbors]).await);
        }
    }
}