use crate::{
    execution::{
        hawk_main::iris_worker::{cache_iris, IrisWorkerPool, QueryId, QuerySpec},
        session::{Session, SessionHandles},
    },
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    hnsw::{
        sorting::{
            min_k_batcher::min_k_batcher_sort_network,
            swap_network::{apply_oblivious_swap_network, SwapNetwork},
        },
        vector_store::VectorStoreMut,
        VectorStore,
    },
    protocol::{
        ops::{
            conditionally_select_distances_with_plain_ids,
            conditionally_select_distances_with_shared_ids, conditionally_swap_distances,
            conditionally_swap_distances_plain_ids, galois_ring_to_rep3, open_ring, DistancePair,
            IdDistance,
        },
        shared_iris::{ArcIris, GaloisRingSharedIris},
    },
    shares::{
        bit::Bit,
        share::{DistanceShare, Share},
        RingElement,
    },
};
use ampc_secret_sharing::shares::{vecshare_bittranspose::Transpose64, VecShare};
use eyre::{bail, OptionExt, Result};
use iris_mpc_common::{iris_db::iris::Threshold, VectorId};
use itertools::{izip, Itertools};
use rand_distr::{Distribution, Standard};
use static_assertions::const_assert;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    sync::Arc,
    vec,
};
use tracing::instrument;

mod distance_fn;
mod distance_ops;
pub use distance_fn::{DistanceFn, DistanceMode};
pub use distance_ops::{DistanceOps, FhdOps, NhdOps};

/// The number of rotations at which to switch from binary tree to round-robin minimum algorithms.
const MIN_ROUND_ROBIN_SIZE: usize = 1;
const_assert!(MIN_ROUND_ROBIN_SIZE >= 1);

/// Lightweight handle referencing a cached query in the `IrisWorkerPool`.
///
/// This is a type alias for `QuerySpec`. The worker pool owns all iris data;
/// `Aby3Query` is just a `(QueryId, rotation, mirrored)` triple that selects
/// a specific preprocessed rotation from the cache.
pub type Aby3Query = QuerySpec;

pub type Aby3DistanceRef<T = u32> = DistanceShare<T>;

pub type Aby3SharedIrises = SharedIrises<ArcIris>;
pub type Aby3SharedIrisesRef = SharedIrisesRef<ArcIris>;

/// Metadata-only VectorId registry — `SharedIrisesRef<()>`.
///
/// Tracks VectorId presence, versions, and checksums without holding iris
/// data.  `Aby3Store` uses this instead of `Aby3SharedIrisesRef` to
/// enforce that all iris data reads go through `IrisWorkerPool`.
pub type VectorIdRegistryRef = SharedIrisesRef<()>;

/// Implementation of VectorStore based on the ABY3 framework (<https://eprint.iacr.org/2018/403.pdf>).
///
/// Generic over `D` (distance operations, e.g. `FhdOps`/`NhdOps`). The worker
/// pool is `Arc<dyn IrisWorkerPool>` so the local and remote sharded pools
/// share one type.
///
/// Note that all SMPC operations are performed in a single session.
#[derive(Debug)]
pub struct Aby3Store<D = FhdOps> {
    /// VectorId registry — tracks presence, versions, and checksums.
    /// Does **not** hold iris data; all iris reads go through `workers`.
    pub registry: VectorIdRegistryRef,

    /// Session for the SMPC operations
    pub session: Session,

    /// Worker pool for CPU-bound distance computations.
    pub workers: Arc<dyn IrisWorkerPool>,

    distance_fn: distance_fn::DistanceFn,

    _phantom: std::marker::PhantomData<D>,
}

impl<D: DistanceOps> Aby3Store<D>
where
    Standard: Distribution<D::Ring>,
    VecShare<D::Ring>: Transpose64,
{
    pub fn new(
        registry: VectorIdRegistryRef,
        session: Session,
        workers: Arc<dyn IrisWorkerPool>,
        distance_mode: DistanceMode,
    ) -> Self {
        Self {
            registry,
            session,
            distance_fn: DistanceFn::new(distance_mode),
            workers,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute pairwise distances between pairs of cached queries.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_pairwise_distances(
        &mut self,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        self.distance_fn.eval_pairwise_distances(self, pairs).await
    }

    /// Converts distances from u16 secret shares to Ring-typed distance shares.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn lift_distances(
        &mut self,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        D::lift_distances(&mut self.session, distances).await
    }

    /// Converts u16 additive sharing (from trick_dot output) to Ring-typed replicated sharing.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn gr_to_lifted_distances(
        &mut self,
        ds_and_ts: Vec<RingElement<u16>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>> {
        let dist = galois_ring_to_rep3(&mut self.session, ds_and_ts).await?;
        self.lift_distances(dist).await
    }

    /// Create a new `Aby3SharedIrises` storage using the specified points mapping.
    pub fn new_storage(points: Option<HashMap<VectorId, ArcIris>>) -> Aby3SharedIrises {
        SharedIrises::new(
            points.unwrap_or_default(),
            Arc::new(GaloisRingSharedIris::default_for_party(0)),
        )
    }

    pub async fn checksum(&self) -> u64 {
        self.registry.checksum().await
    }

    /// Fetch a stored vector's iris from the worker pool and cache it as a query.
    /// Returns a query handle (center rotation, non-mirrored).
    pub async fn cache_query_from_store(&self, vector: &VectorId) -> Result<Aby3Query> {
        let irises = self.workers.fetch_irises(vec![*vector]).await?;
        let iris = irises
            .into_iter()
            .next()
            .ok_or_eyre("fetch_irises did not return expected iris or empty default")?;
        cache_iris(self.workers.as_ref(), iris).await
    }

    /// Obliviously swaps the elements in `list` at the given `indices` according to the `swap_bits`.
    /// If bit is 0, the elements are swapped, otherwise they are left unchanged.
    /// Note that unchanged elements of the list are propagated as secret-shares.
    pub async fn oblivious_swap_batch_plain_ids(
        &mut self,
        swap_bits: Vec<Share<Bit>>,
        list: &[(u32, DistanceShare<D::Ring>)],
        indices: &[(usize, usize)],
    ) -> Result<Vec<(Share<D::Ring>, DistanceShare<D::Ring>)>> {
        if list.is_empty() {
            return Ok(vec![]);
        }

        conditionally_swap_distances_plain_ids(&mut self.session, swap_bits, list, indices).await
    }

    /// Obliviously compares pairs of distances in batch and returns a secret shared bit a < b for each pair.
    pub async fn oblivious_less_than_batch(
        &mut self,
        distances: &[DistancePair<D::Ring>],
    ) -> Result<Vec<Share<Bit>>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        D::oblivious_cross_compare(&mut self.session, distances).await
    }

    /// Obliviously swaps the elements in `list` at the given `indices` according to the `swap_bits`.
    /// If bit is 0, the elements are swapped, otherwise they are left unchanged.
    pub async fn oblivious_swap_batch(
        &mut self,
        swap_bits: Vec<Share<Bit>>,
        list: &[IdDistance<D::Ring>],
        indices: &[(usize, usize)],
    ) -> Result<Vec<IdDistance<D::Ring>>> {
        if list.is_empty() {
            return Ok(vec![]);
        }

        conditionally_swap_distances(&mut self.session, swap_bits, list, indices).await
    }

    /// Obliviously computes the minimum distance of a given distance array.
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    pub async fn oblivious_min_distance(
        &mut self,
        distances: &[DistanceShare<D::Ring>],
    ) -> Result<DistanceShare<D::Ring>>
    where
        VecShare<D::Ring>: Transpose64,
    {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        if distances.len() == 1 {
            return Ok(distances[0]);
        }
        let mut res = distances.to_vec();
        while res.len() > MIN_ROUND_ROBIN_SIZE {
            // if the length is odd, we save the last distance to add it back later
            let maybe_last_distance = if res.len() % 2 == 1 { res.pop() } else { None };
            // create pairs from the remaining distances
            let pairs: Vec<(_, _)> = res.into_iter().tuples().collect_vec();
            // compute minimums of pairs
            res = D::min_of_pair_batch(&mut self.session, &pairs).await?;
            // if we saved a last distance, we need to add it back
            if let Some(last_distance) = maybe_last_distance {
                res.push(last_distance);
            }
        }
        D::min_round_robin_batch(&mut self.session, &res, res.len())
            .await?
            .pop()
            .ok_or_eyre("Should not be here: distances are empty")
    }

    /// Obliviously computes the minimum distance and the corresponding vector id of a given array of pairs (id, distance).
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    pub async fn oblivious_argmin_distance(
        &mut self,
        distances: &[(VectorId, DistanceShare<D::Ring>)],
    ) -> Result<(VectorId, DistanceShare<D::Ring>)> {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        if distances.len() == 1 {
            return Ok(distances[0]);
        }

        // Handle plain ids first
        let mut plain_res = distances
            .iter()
            .enumerate()
            .map(|(id, (_, distance))| (id as u32, *distance))
            .collect_vec();
        let plain_maybe_last_distance = if plain_res.len() % 2 == 1 {
            plain_res.pop()
        } else {
            None
        };
        let mut dist_pairs = plain_res
            .iter()
            .tuples()
            .map(|((_, dist1), (_, dist2))| (*dist1, *dist2))
            .collect_vec();
        let mut control_bits =
            D::oblivious_cross_compare_lifted(&mut self.session, &dist_pairs).await?;
        let (left_dist, right_dist) = plain_res.into_iter().tuples().unzip();
        let mut res = conditionally_select_distances_with_plain_ids(
            &mut self.session,
            left_dist,
            right_dist,
            control_bits,
        )
        .await?;
        // If we saved a last distance, we need to add it back
        if let Some((id, dist)) = plain_maybe_last_distance {
            let shared_id = Share::from_const(D::Ring::from(id), self.session.own_role());
            res.push((shared_id, dist));
        }

        // Now handle distances with shared ids
        while res.len() > 1 {
            // if the length is odd, we save the last distance to add it back later
            let maybe_last_distance = if res.len() % 2 == 1 { res.pop() } else { None };
            // create pairs from the remaining distances
            dist_pairs = res
                .iter()
                .tuples()
                .map(|((_, dist1), (_, dist2))| (*dist1, *dist2))
                .collect_vec();
            // compute minimums of pairs
            control_bits =
                D::oblivious_cross_compare_lifted(&mut self.session, &dist_pairs).await?;
            let (left_dist, right_dist) = res.into_iter().tuples().unzip();
            res = conditionally_select_distances_with_shared_ids(
                &mut self.session,
                left_dist,
                right_dist,
                control_bits,
            )
            .await?;
            // if we saved a last distance, we need to add it back
            if let Some(dist) = maybe_last_distance {
                res.push(dist);
            }
        }
        // res is guaranteed to have length 1
        let (shared_id, dist) = res
            .pop()
            .ok_or_eyre("Shouldn't be here: results are empty")?;
        // open the id
        let id = open_ring(&mut self.session, &[shared_id]).await?[0];
        let res = (distances[D::to_usize(id)].0, dist);
        Ok(res)
    }

    /// Obliviously computes the minimum distance for each batch of given distances of the same size.
    /// The input `distances` is a 2D matrix with dimensions: (rotations, batch).
    /// `distances[r][i]` corresponds to the rth rotation of the ith item of the batch.
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    pub(crate) async fn oblivious_min_distance_batch(
        &mut self,
        distances: Vec<Vec<DistanceShare<D::Ring>>>,
    ) -> Result<Vec<DistanceShare<D::Ring>>>
    where
        VecShare<D::Ring>: Transpose64,
    {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        let len = distances[0].len();
        for (i, d) in distances.iter().enumerate() {
            if d.len() != len {
                eyre::bail!("All distance lists must have the same length. List at index {} has length {}, while the first list has length {}", i, d.len(), len);
            }
        }

        let mut res = distances;
        let mut pairs = Vec::with_capacity(len * (res.len() / 2));
        while res.len() > MIN_ROUND_ROBIN_SIZE {
            // if the length is odd, we save the last distance to add it back later
            let maybe_last_distance = if res.len() % 2 == 1 { res.pop() } else { None };

            // Build pairs for min_of_pair_batch
            pairs.clear();
            for ab in res.chunks_exact(2) {
                let (a, b) = (&ab[0], &ab[1]);
                for (x, y) in izip!(a, b) {
                    pairs.push((*x, *y));
                }
            }

            // compute minimums of pairs
            let flattened_res = D::min_of_pair_batch(&mut self.session, &pairs).await?;

            // Rebuild res as Vec<Vec<_>>
            res.clear();
            for chunk in flattened_res.chunks(len) {
                res.push(chunk.to_vec());
            }
            // if we saved a last distance, we need to add it back
            if let Some(last_distance) = maybe_last_distance {
                res.push(last_distance);
            }
        }
        // Only flatten res once at the end
        let res_len = res.len();
        let mut flattened_distances = Vec::with_capacity(res_len * len);
        flattened_distances.extend(res.into_iter().flatten());
        D::min_round_robin_batch(&mut self.session, &flattened_distances, res_len).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn compact_neighborhood_batch(
        &mut self,
        base_nodes: &[VectorId],
        neighborhoods: &[Vec<VectorId>],
        max_sizes: &[usize],
    ) -> Result<Vec<Vec<VectorId>>> {
        if base_nodes.len() != neighborhoods.len() || base_nodes.len() != max_sizes.len() {
            bail!("Lists of base nodes, neighborhoods, and max sizes must have equal sizes");
        }

        let base_node_queries = self.vectors_as_queries(base_nodes.to_vec()).await?;
        let cached_qids: Vec<QueryId> = base_node_queries.iter().map(|q| q.query_id).collect();
        let batches: Vec<(Aby3Query, Vec<VectorId>)> =
            izip!(base_node_queries, neighborhoods.iter())
                .map(|(q, nbhd)| (q, nbhd.clone()))
                .collect();
        let distance_start = std::time::Instant::now();
        let nbhd_distances = self.eval_distance_multibatch(batches).await?;
        metrics::histogram!("compaction_distance_duration")
            .record(distance_start.elapsed().as_secs_f64());
        let id_distances = izip!(
            neighborhoods.iter().flatten().map(|vid| vid.serial_id()),
            nbhd_distances.into_iter().flatten(),
        )
        .collect_vec();
        let id_versions: BTreeMap<_, _> = neighborhoods
            .iter()
            .enumerate()
            .flat_map(|(idx, nbhd)| {
                nbhd.iter()
                    .map(move |vector_id| ((idx, vector_id.serial_id()), vector_id.version_id()))
            })
            .collect();

        // Construct aggregated selection networks for top-k selection over all neighborhoods
        let mut total_items: usize = 0;
        let mut batched_network = SwapNetwork::new();
        for (nbhd, target_size) in izip!(neighborhoods.iter(), max_sizes.iter()) {
            let current_size = nbhd.len();

            // Constructed network is already optimized for the case of k > n - k
            let network = min_k_batcher_sort_network(current_size, *target_size)?;

            // Merge individual swap network into overall batch network
            let network_shift_amount = isize::try_from(total_items)?;
            batched_network.insert_parallel_in_place(network, network_shift_amount)?;

            total_items += current_size;
        }

        // Oblivious application of batched selection networks
        let sorting_start = std::time::Instant::now();
        let res_id_distances =
            apply_oblivious_swap_network(self, &id_distances, &batched_network).await?;
        metrics::histogram!("compaction_sorting_duration")
            .record(sorting_start.elapsed().as_secs_f64());

        // Truncate results and unpack into individual vectors
        let mut unshuffled_truncated_shares = Vec::with_capacity(neighborhoods.len());
        let mut base_idx = 0;
        for (nbhd, max_size) in izip!(neighborhoods.iter(), max_sizes.iter()) {
            let n_keep = usize::min(nbhd.len(), *max_size);
            let nbhd_shares = res_id_distances[base_idx..base_idx + n_keep].to_vec();
            unshuffled_truncated_shares.push(nbhd_shares);
            base_idx += nbhd.len();
        }

        // Organize vectors by length for batch shuffling. (Batched shuffle
        // protocol implementation is currently limited to a single list length
        // over the batch.)
        let mut shares_by_length: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for (idx, nbhd_shares) in unshuffled_truncated_shares.into_iter().enumerate() {
            let v = shares_by_length.entry(nbhd_shares.len()).or_default();
            v.push((idx, nbhd_shares));
        }

        // Batch shuffle
        let shuffle_start = std::time::Instant::now();
        let mut shuffled_shares_by_idx: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for (_len, v) in shares_by_length.into_iter() {
            let (idxs, nbhds): (Vec<_>, Vec<_>) = v.into_iter().unzip();

            let shuffled_nbhds = D::shuffle_batch(&mut self.session, nbhds).await?;

            for (idx, shuffled_nbhd) in izip!(idxs, shuffled_nbhds) {
                shuffled_shares_by_idx.insert(idx, shuffled_nbhd);
            }
        }
        metrics::histogram!("compaction_shuffle_duration")
            .record(shuffle_start.elapsed().as_secs_f64());

        // Open secret shared neighborhood vector ids
        let secret_nbhds = shuffled_shares_by_idx
            .into_values()
            .map(|nbhd| {
                nbhd.into_iter()
                    .map(|(idx_share, _dist_share)| idx_share)
                    .collect_vec()
            })
            .collect_vec();
        let nbhd_lengths = secret_nbhds.iter().map(|n| n.len()).collect_vec();
        let opened_nbhds_flat = open_ring(
            &mut self.session,
            &secret_nbhds.into_iter().flatten().collect_vec(),
        )
        .await?;

        // Unflatten opened neighborhoods
        let mut nbhd_serial_ids = Vec::with_capacity(neighborhoods.len());
        let mut base_idx = 0;
        for len in nbhd_lengths {
            let opened_nbhd = opened_nbhds_flat[base_idx..base_idx + len].to_vec();
            nbhd_serial_ids.push(opened_nbhd);
            base_idx += len;
        }

        // Reconstruct versions of vector ids
        let compacted_nbhds = nbhd_serial_ids
            .into_iter()
            .enumerate()
            .map(|(idx, nbhd)| {
                nbhd.into_iter()
                    .map(|serial_id| {
                        let serial_id_u32 = D::to_usize(serial_id) as u32;
                        let version =
                            *id_versions.get(&(idx, serial_id_u32)).ok_or_eyre(format!(
                                "Unexpected: found no record of reconstructed serial id: {}",
                                serial_id_u32
                            ))?;
                        Ok(VectorId::new(serial_id_u32, version))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        // Evict cached queries from vectors_as_queries now that all
        // distance computation using them is complete.
        self.workers.evict_queries(cached_qids).await?;

        Ok(compacted_nbhds)
    }

    /// Evaluates distances for multiple (query, vectors) batches.
    ///
    /// Optimized for MinRotation where prerotation buffer is reused per query.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_distance_multibatch(
        &mut self,
        batches: Vec<(Aby3Query, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<DistanceShare<D::Ring>>>> {
        if batches.is_empty() {
            return Ok(vec![]);
        }
        self.distance_fn
            .eval_distance_multibatch(self, batches)
            .await
    }

    /// Check whether a batch of distances are matches at the given threshold.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn is_match_at(
        &mut self,
        distances: &[DistanceShare<D::Ring>],
        threshold: Threshold,
    ) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        D::lte_and_open(&mut self.session, distances, threshold).await
    }
}

impl<D: DistanceOps> VectorStore for Aby3Store<D>
where
    Standard: Distribution<D::Ring>,
    VecShare<D::Ring>: Transpose64,
{
    /// Arc ref to a query.
    type QueryRef = Aby3Query;
    /// Distance represented as a pair of Ring-typed shares.
    type DistanceRef = Aby3DistanceRef<D::Ring>;

    async fn vectors_as_queries(&mut self, vectors: Vec<VectorId>) -> Result<Vec<Self::QueryRef>> {
        let irises = self.workers.fetch_irises(vectors).await?;
        let to_cache: Vec<_> = irises
            .into_iter()
            .map(|iris| (QueryId::new(), iris))
            .collect();
        let query_ids: Vec<QueryId> = to_cache.iter().map(|(qid, _)| *qid).collect();
        self.workers.cache_queries(to_cache).await?;
        Ok(query_ids.into_iter().map(Aby3Query::new).collect_vec())
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &VectorId,
    ) -> Result<Self::DistanceRef> {
        let mut d = self.eval_distance_batch(query, &[*vector]).await?;
        d.pop()
            .ok_or_eyre("eval_distance_batch did not return expected distance")
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(queries = pairs.len(), batch_size = pairs.len()))]
    async fn eval_distance_pairs(
        &mut self,
        pairs: &[(Self::QueryRef, VectorId)],
    ) -> Result<Vec<Self::DistanceRef>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }
        self.distance_fn.eval_distance_pairs(self, pairs).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = vectors.len()))]
    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[VectorId],
    ) -> Result<Vec<Self::DistanceRef>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        metrics::counter!("distance_evaluations_total").increment(vectors.len() as u64);
        metrics::histogram!("distance_evaluations_batch_size").record(vectors.len() as f64);
        let start = std::time::Instant::now();
        let result = self
            .distance_fn
            .eval_distance_batch(self, query, vectors)
            .await;
        metrics::histogram!("eval_distance_batch_duration").record(start.elapsed().as_secs_f64());
        result
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn get_argmin_distance(
        &mut self,
        distances: &[(VectorId, Self::DistanceRef)],
    ) -> Result<(VectorId, Self::DistanceRef)> {
        if distances.is_empty() {
            return Err(eyre::eyre!("Cannot get min of empty list"));
        }
        self.oblivious_argmin_distance(distances).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(D::lte_and_open(
            &mut self.session,
            std::slice::from_ref(distance),
            Threshold::Match,
        )
        .await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        Ok(D::cross_compare(&mut self.session, &[(*distance1, *distance2)]).await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        metrics::counter!("comparisons_total").increment(distances.len() as u64);
        metrics::histogram!("comparisons_batch_size").record(distances.len() as f64);
        let start = std::time::Instant::now();
        let result = D::cross_compare(&mut self.session, distances).await;
        metrics::histogram!("less_than_batch_duration").record(start.elapsed().as_secs_f64());
        result
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn is_match_batch(&mut self, distances: &[Self::DistanceRef]) -> Result<Vec<bool>> {
        self.is_match_at(distances, Threshold::Match).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn compact_neighborhood(
        &mut self,
        base_node: VectorId,
        neighborhood: &[VectorId],
        max_size: usize,
    ) -> Result<Vec<VectorId>> {
        let compaction_list = self
            .compact_neighborhood_batch(&[base_node], &[neighborhood.to_vec()], &[max_size])
            .await?;
        compaction_list
            .first()
            .ok_or_eyre("Unexpected: no compacted neighborhood returned from batch processing")
            .cloned()
    }

    async fn compact_neighborhood_batch(
        &mut self,
        base_nodes: &[VectorId],
        neighborhoods: &[Vec<VectorId>],
        max_sizes: &[usize],
    ) -> Result<Vec<Vec<VectorId>>> {
        self.compact_neighborhood_batch(base_nodes, neighborhoods, max_sizes)
            .await
    }
}

impl<D: DistanceOps> VectorStoreMut for Aby3Store<D>
where
    Standard: Distribution<D::Ring>,
    VecShare<D::Ring>: Transpose64,
{
    async fn insert(&mut self, query: &Self::QueryRef) -> VectorId {
        // Allocate next ID and register it in the registry (metadata only).
        let vector_id = {
            let mut reg = self.registry.write().await;
            let id = reg.allocate_next_id();
            reg.insert(id, ());
            id
        };
        // Insert the actual iris data into the worker's store.
        self.workers
            .insert_irises(vec![(query.query_id, vector_id)])
            .await
            .expect("insert_irises failed: query not cached or store write failed");
        vector_id
    }

    async fn insert_at(
        &mut self,
        vector_ref: &VectorId,
        query: &Self::QueryRef,
    ) -> Result<VectorId> {
        // Register in the metadata registry.
        self.registry.write().await.insert(*vector_ref, ());
        // Insert the actual iris data into the worker's store.
        self.workers
            .insert_irises(vec![(query.query_id, *vector_ref)])
            .await?;
        Ok(*vector_ref)
    }
}

#[cfg(test)]
mod tests;
