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
#[cfg(test)]
use ampc_actor_utils::protocol::fhd_ops::fhd_greater_than_anon_stats_threshold;
use ampc_actor_utils::protocol::{
    binary::open_bin,
    fhd_ops::{
        fhd_greater_than_anon_stats_from_galois, fhd_greater_than_threshold_pre_lifted_masks,
        lift_fhd_mask_dots,
    },
    ops::batch_signed_lift_vec,
};
use ampc_secret_sharing::shares::{vecshare_bittranspose::Transpose64, VecShare};
use eyre::{bail, OptionExt, Result};
use iris_mpc_common::{iris_db::iris::Threshold, VectorId, ROTATIONS};
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
pub type RotationMatchIndices = Vec<Vec<usize>>;

/// Both orientations' additive dot-product shares for one chunk.
pub type PairDotContributions = [Vec<RingElement<u16>>; 2];

/// A spawned task handle that aborts the task when dropped, so a lookahead
/// chunk cannot keep running detached after the lane that requested it has
/// failed or been cancelled.
pub struct AbortOnDropHandle<T>(tokio::task::JoinHandle<T>);

impl<T> AbortOnDropHandle<T> {
    fn new(handle: tokio::task::JoinHandle<T>) -> Self {
        Self(handle)
    }
}

impl<T> Drop for AbortOnDropHandle<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

impl<T> std::future::Future for AbortOnDropHandle<T> {
    type Output = std::result::Result<T, tokio::task::JoinError>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // `JoinHandle` is `Unpin`, so projecting through the wrapper is sound.
        std::pin::Pin::new(&mut self.get_mut().0).poll(cx)
    }
}

/// GPU-equivalent exact-scan classification for one chunk. Thresholds are
/// evaluated directly for all 31 rotations; no secret minimum is computed.
#[derive(Debug)]
pub struct FullRotationThresholdResult {
    /// One representative matching rotation distance per vector.
    pub matches: Vec<Option<DistanceShare<u32>>>,
    /// Every rotation distance passing the wider anonymous-statistics threshold,
    /// represented as `(vector index in the input chunk, rotation, distance)`.
    pub anon_stats_matches: Vec<(usize, usize, DistanceShare<u32>)>,
    /// Rotation indices passing the strict match threshold, grouped by vector.
    pub match_rotations: RotationMatchIndices,
}

/// Expand records passing the public anonymous-statistics prefilter back to
/// all database rotations for the strict comparison pass.
///
/// The GPU actor first ORs the opened per-rotation bits into a record bitmap,
/// then evaluates all 31 rotations of every record in that bitmap. Keeping the
/// same expansion here preserves its protocol transcript as well as its result
/// semantics.
fn gpu_candidate_rotation_indices(anon_rotation_bits: &[bool]) -> Vec<usize> {
    debug_assert!(anon_rotation_bits.len().is_multiple_of(ROTATIONS));
    anon_rotation_bits
        .chunks_exact(ROTATIONS)
        .enumerate()
        .filter_map(|(vector, rotations)| rotations.iter().any(|&bit| bit).then_some(vector))
        .flat_map(|vector| {
            let start = vector * ROTATIONS;
            start..start + ROTATIONS
        })
        .collect()
}

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

    /// Full-rotation exact scan with the opened per-rotation match metadata
    /// retained as a correctness oracle for the fused production path.
    #[cfg(test)]
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_distance_batch_full_rotations_with_rotation_matches(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<(Vec<DistanceShare<D::Ring>>, RotationMatchIndices)> {
        if vectors.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let rotation_distances = self.full_rotation_distances(query, vectors).await?;
        let distances = self
            .oblivious_min_distance_batch(distance_fn::transpose_from_flat_with_rotations(
                &rotation_distances,
                ROTATIONS,
            ))
            .await?;
        let rotation_match_bits =
            D::lte_and_open(&mut self.session, &rotation_distances, Threshold::Match).await?;
        let rotation_matches = rotation_match_bits
            .chunks(ROTATIONS)
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .filter_map(|(rotation, &is_match)| is_match.then_some(rotation))
                    .collect()
            })
            .collect();
        Ok((distances, rotation_matches))
    }

    #[cfg(test)]
    async fn full_rotation_distances(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<DistanceShare<D::Ring>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        let dot_shares = self.full_rotation_dot_shares(query, vectors).await?;
        self.lift_distances(dot_shares).await
    }

    #[cfg(test)]
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn full_rotation_dot_shares(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<Share<u16>>> {
        let ds_and_ts = self.full_rotation_dot_contributions(query, vectors).await?;
        galois_ring_to_rep3(&mut self.session, ds_and_ts).await
    }

    /// Compute the local additive dot-product contributions before refreshing
    /// them into replicated shares. The fused exact-scan path consumes this
    /// representation directly and materializes scalar shares only for public
    /// candidates.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn full_rotation_dot_contributions(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<Vec<RingElement<u16>>> {
        // This scan is neither the simple nor the min-rotation distance: it
        // opens a threshold for each of the 31 rotations separately. What it
        // does rely on is the Hawk query layout, where the cached query
        // rotations are addressed relative to `CENTER_ROTATION`; the worker
        // derives all database rotations from that single center query.
        eyre::ensure!(
            query.rotation == crate::execution::hawk_main::iris_worker::CENTER_ROTATION,
            "full-rotation scan must start from the center query rotation"
        );
        metrics::counter!("distance_evaluations_total").increment(vectors.len() as u64);
        metrics::histogram!("distance_evaluations_batch_size").record(vectors.len() as f64);
        self.workers
            .compute_dot_products_full_rotations(*query, vectors.to_vec())
            .await
    }

    /// Dispatch both orientations' local dot contributions for one chunk as a
    /// spawned task on the worker pool. The caller can drive the previous
    /// chunk's threshold rounds while this chunk's dot products compute,
    /// keeping the dot workers fed. Each returned side is identical to a
    /// separate [`Self::full_rotation_dot_contributions`] call; only the
    /// worker-level target streaming is shared. This performs no network
    /// communication, so fusing and pipelining the dot passes is invisible to
    /// the MPC transcript.
    ///
    /// Configuration errors are reported before anything is spawned. The
    /// returned handle aborts the task when dropped, so a lane that fails
    /// while a lookahead chunk is in flight does not leave that chunk running
    /// detached on the dot-product workers.
    pub fn spawn_full_rotation_dot_contributions_pair(
        &self,
        queries: [&Aby3Query; 2],
        vectors: &[VectorId],
    ) -> Result<AbortOnDropHandle<Result<PairDotContributions>>> {
        // See `full_rotation_dot_contributions`: only the center-rotation
        // query layout matters, not the configured distance mode.
        for query in queries {
            eyre::ensure!(
                query.rotation == crate::execution::hawk_main::iris_worker::CENTER_ROTATION,
                "full-rotation scan must start from the center query rotation"
            );
        }
        let specs = [*queries[0], *queries[1]];
        let workers = self.workers.clone();
        let vectors = vectors.to_vec();
        Ok(AbortOnDropHandle::new(tokio::spawn(async move {
            metrics::counter!("distance_evaluations_total").increment(2 * vectors.len() as u64);
            metrics::histogram!("distance_evaluations_batch_size").record(vectors.len() as f64);
            workers
                .compute_dot_products_full_rotations_pair(specs, vectors)
                .await
        })))
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

impl Aby3Store<FhdOps> {
    /// Unfused exact-scan threshold implementation retained as a correctness
    /// oracle for the production path.
    #[cfg(test)]
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_distance_batch_full_rotation_thresholds(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<FullRotationThresholdResult> {
        if vectors.is_empty() {
            return Ok(FullRotationThresholdResult {
                matches: Vec::new(),
                anon_stats_matches: Vec::new(),
                match_rotations: Vec::new(),
            });
        }

        let dot_shares = self.full_rotation_dot_shares(query, vectors).await?;
        let expected_dots = vectors.len() * ROTATIONS * 2;
        eyre::ensure!(
            dot_shares.len() == expected_dots,
            "full-rotation dot result has unexpected length"
        );
        let code_dots = dot_shares.iter().step_by(2).copied().collect::<Vec<_>>();
        let mask_dots = dot_shares
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>();

        // The fixed anonymous-statistics threshold is evaluated directly in
        // an 18-bit binary circuit. Do not lift masks for the full chunk: only
        // public candidates need arithmetic distances or the strict threshold.
        let anon_gt =
            fhd_greater_than_anon_stats_threshold(&mut self.session, &code_dots, &mask_dots)
                .await?;
        let anon_rotation_bits = open_bin(&mut self.session, &anon_gt)
            .await?
            .into_iter()
            .map(|bit| !bool::from(bit))
            .collect::<Vec<_>>();

        eyre::ensure!(
            anon_rotation_bits.len() == code_dots.len(),
            "anonymous threshold result has unexpected length"
        );

        // Exactly like the GPU actor, collapse the wider public prefilter to a
        // record bitmap, then run the strict threshold over all 31 rotations
        // of every surviving record. Although the strict threshold is a subset
        // of the anonymous threshold and per-rotation filtering would produce
        // the same matches, it would change observable protocol message sizes.
        let candidate_rotation_indices = gpu_candidate_rotation_indices(&anon_rotation_bits);
        let candidate_codes = candidate_rotation_indices
            .iter()
            .map(|&index| code_dots[index])
            .collect::<Vec<_>>();
        let candidate_raw_masks = candidate_rotation_indices
            .iter()
            .map(|&index| mask_dots[index])
            .collect::<Vec<_>>();
        let candidate_lifted_masks = if candidate_raw_masks.is_empty() {
            Vec::new()
        } else {
            lift_fhd_mask_dots(&mut self.session, &candidate_raw_masks).await?
        };
        let mut match_rotation_bits = vec![false; code_dots.len()];
        if !candidate_rotation_indices.is_empty() {
            let match_gt = fhd_greater_than_threshold_pre_lifted_masks(
                &mut self.session,
                &candidate_codes,
                &candidate_lifted_masks,
                Threshold::Match.ratio(),
            )
            .await?;
            let candidate_match_bits = open_bin(&mut self.session, &match_gt)
                .await?
                .into_iter()
                .map(|bit| !bool::from(bit));
            for (&index, is_match) in candidate_rotation_indices.iter().zip(candidate_match_bits) {
                match_rotation_bits[index] = is_match;
            }
        }

        eyre::ensure!(
            match_rotation_bits
                .iter()
                .zip(&anon_rotation_bits)
                .all(|(&is_match, &is_anon_match)| !is_match || is_anon_match),
            "strict match threshold produced a result outside the anonymous prefilter"
        );

        // GPU anonymous-statistics persistence retains the actual distance of
        // each passing rotation. Code lifts are needed only for those passing
        // rotations; mask lifts above cover all rotations of candidate records.
        let anon_rotation_indices = anon_rotation_bits
            .iter()
            .enumerate()
            .filter_map(|(index, &is_match)| is_match.then_some(index))
            .collect::<Vec<_>>();
        let anon_codes = anon_rotation_indices
            .iter()
            .map(|&index| code_dots[index])
            .collect::<Vec<_>>();
        let lifted_anon_codes = if anon_codes.is_empty() {
            Vec::new()
        } else {
            batch_signed_lift_vec(&mut self.session, anon_codes).await?
        };

        let mut matches = vec![None; vectors.len()];
        let mut anon_stats_matches = Vec::with_capacity(anon_rotation_indices.len());
        for (&index, code_dot) in anon_rotation_indices.iter().zip(lifted_anon_codes) {
            let vector = index / ROTATIONS;
            let rotation = index % ROTATIONS;
            let candidate_index = candidate_rotation_indices
                .binary_search(&index)
                .expect("anonymous rotation must belong to a candidate record");
            let distance = DistanceShare::new(code_dot, candidate_lifted_masks[candidate_index]);
            anon_stats_matches.push((vector, rotation, distance));
            if match_rotation_bits[index] && matches[vector].is_none() {
                matches[vector] = Some(distance);
            }
        }

        let match_rotations = match_rotation_bits
            .chunks_exact(ROTATIONS)
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .filter_map(|(rotation, &is_match)| is_match.then_some(rotation))
                    .collect()
            })
            .collect();

        Ok(FullRotationThresholdResult {
            matches,
            anon_stats_matches,
            match_rotations,
        })
    }

    /// Allocation-fused threshold implementation used by the production CPU
    /// exact scan.
    ///
    /// It preserves the Galois-to-Rep3 refresh and the threshold circuit's
    /// network transcript, but bit-transposes the two refreshed components
    /// directly instead of first allocating a dense scalar `Share<u16>` batch
    /// and three mostly-zero packed component vectors.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_distance_batch_full_rotation_thresholds_fused(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
    ) -> Result<FullRotationThresholdResult> {
        self.eval_distance_batch_full_rotation_thresholds_fused_with_forced_anon_stats(
            query,
            vectors,
            &[],
        )
        .await
    }

    /// GPU-compatible fused scan with selected records retained for anonymous
    /// statistics regardless of the anonymous-statistics threshold. The CUDA
    /// actor uses this for reauthentication targets.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_distance_batch_full_rotation_thresholds_fused_with_forced_anon_stats(
        &mut self,
        query: &Aby3Query,
        vectors: &[VectorId],
        forced_anon_stats_vectors: &[usize],
    ) -> Result<FullRotationThresholdResult> {
        if vectors.is_empty() {
            return Ok(FullRotationThresholdResult {
                matches: Vec::new(),
                anon_stats_matches: Vec::new(),
                match_rotations: Vec::new(),
            });
        }

        let dot_contributions = self.full_rotation_dot_contributions(query, vectors).await?;
        self.eval_full_rotation_thresholds_fused_from_contributions_with_forced_anon_stats(
            dot_contributions,
            vectors.len(),
            forced_anon_stats_vectors,
        )
        .await
    }

    /// Threshold half of [`Self::eval_distance_batch_full_rotation_thresholds_fused`],
    /// taking precomputed full-rotation dot contributions for `n_vectors`
    /// records. The dot phase is pure local worker compute, so callers can
    /// overlap the next chunk's dot products with this chunk's threshold
    /// network rounds without changing the per-session wire transcript.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_full_rotation_thresholds_fused_from_contributions(
        &mut self,
        dot_contributions: Vec<RingElement<u16>>,
        n_vectors: usize,
    ) -> Result<FullRotationThresholdResult> {
        self.eval_full_rotation_thresholds_fused_from_contributions_with_forced_anon_stats(
            dot_contributions,
            n_vectors,
            &[],
        )
        .await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_full_rotation_thresholds_fused_from_contributions_with_forced_anon_stats(
        &mut self,
        dot_contributions: Vec<RingElement<u16>>,
        n_vectors: usize,
        forced_anon_stats_vectors: &[usize],
    ) -> Result<FullRotationThresholdResult> {
        if n_vectors == 0 {
            return Ok(FullRotationThresholdResult {
                matches: Vec::new(),
                anon_stats_matches: Vec::new(),
                match_rotations: Vec::new(),
            });
        }
        let vectors_len = n_vectors;
        let expected_dots = vectors_len * ROTATIONS * 2;
        eyre::ensure!(
            dot_contributions.len() == expected_dots,
            "full-rotation dot result has unexpected length"
        );
        let (anon_gt, dot_shares) =
            fhd_greater_than_anon_stats_from_galois(&mut self.session, dot_contributions).await?;
        eyre::ensure!(
            dot_shares.len() == vectors_len * ROTATIONS,
            "fused full-rotation dot result has unexpected length"
        );
        let mut anon_rotation_bits = open_bin(&mut self.session, &anon_gt)
            .await?
            .into_iter()
            .map(|bit| !bool::from(bit))
            .collect::<Vec<_>>();

        eyre::ensure!(
            anon_rotation_bits.len() == dot_shares.len(),
            "anonymous threshold result has unexpected length"
        );

        // CUDA unions the reauthentication target into the public candidate
        // bitmap and stores all of its rotations, even those outside the
        // anonymous-statistics threshold.
        for &vector in forced_anon_stats_vectors {
            eyre::ensure!(
                vector < vectors_len,
                "forced anonymous-statistics vector index is out of bounds"
            );
            anon_rotation_bits[vector * ROTATIONS..(vector + 1) * ROTATIONS].fill(true);
        }

        let dot_count = dot_shares.len();
        let candidate_rotation_indices = gpu_candidate_rotation_indices(&anon_rotation_bits);
        let (candidate_codes, candidate_raw_masks) =
            dot_shares.select(&candidate_rotation_indices)?;
        drop(dot_shares);
        let candidate_lifted_masks = if candidate_raw_masks.is_empty() {
            Vec::new()
        } else {
            lift_fhd_mask_dots(&mut self.session, &candidate_raw_masks).await?
        };
        let mut match_rotation_bits = vec![false; dot_count];
        if !candidate_rotation_indices.is_empty() {
            let match_gt = fhd_greater_than_threshold_pre_lifted_masks(
                &mut self.session,
                &candidate_codes,
                &candidate_lifted_masks,
                Threshold::Match.ratio(),
            )
            .await?;
            let candidate_match_bits = open_bin(&mut self.session, &match_gt)
                .await?
                .into_iter()
                .map(|bit| !bool::from(bit));
            for (&index, is_match) in candidate_rotation_indices.iter().zip(candidate_match_bits) {
                match_rotation_bits[index] = is_match;
            }
        }

        eyre::ensure!(
            match_rotation_bits
                .iter()
                .zip(&anon_rotation_bits)
                .all(|(&is_match, &is_anon_match)| !is_match || is_anon_match),
            "strict match threshold produced a result outside the anonymous prefilter"
        );

        let anon_rotation_indices = anon_rotation_bits
            .iter()
            .enumerate()
            .filter_map(|(index, &is_match)| is_match.then_some(index))
            .collect::<Vec<_>>();
        let anon_codes = anon_rotation_indices
            .iter()
            .map(|index| {
                let candidate_index = candidate_rotation_indices
                    .binary_search(index)
                    .expect("anonymous rotation must belong to a candidate record");
                candidate_codes[candidate_index]
            })
            .collect::<Vec<_>>();
        let lifted_anon_codes = if anon_codes.is_empty() {
            Vec::new()
        } else {
            batch_signed_lift_vec(&mut self.session, anon_codes).await?
        };

        let mut matches = vec![None; vectors_len];
        let mut anon_stats_matches = Vec::with_capacity(anon_rotation_indices.len());
        for (&index, code_dot) in anon_rotation_indices.iter().zip(lifted_anon_codes) {
            let vector = index / ROTATIONS;
            let rotation = index % ROTATIONS;
            let candidate_index = candidate_rotation_indices
                .binary_search(&index)
                .expect("anonymous rotation must belong to a candidate record");
            let distance = DistanceShare::new(code_dot, candidate_lifted_masks[candidate_index]);
            anon_stats_matches.push((vector, rotation, distance));
            if match_rotation_bits[index] && matches[vector].is_none() {
                matches[vector] = Some(distance);
            }
        }

        let match_rotations = match_rotation_bits
            .chunks_exact(ROTATIONS)
            .map(|bits| {
                bits.iter()
                    .enumerate()
                    .filter_map(|(rotation, &is_match)| is_match.then_some(rotation))
                    .collect()
            })
            .collect();

        Ok(FullRotationThresholdResult {
            matches,
            anon_stats_matches,
            match_rotations,
        })
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
