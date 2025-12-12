use crate::{
    execution::{
        hawk_main::iris_worker::IrisPoolHandle,
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
            batch_signed_lift_vec, conditionally_select_distances_with_plain_ids,
            conditionally_select_distances_with_shared_ids, conditionally_swap_distances,
            conditionally_swap_distances_plain_ids, cross_compare, galois_ring_to_rep3,
            lte_threshold_and_open, min_of_pair_batch, min_round_robin_batch,
            oblivious_cross_compare, oblivious_cross_compare_lifted, open_ring,
        },
        shared_iris::{ArcIris, GaloisRingSharedIris},
    },
    shares::{
        bit::Bit,
        share::{DistanceShare, Share},
        RingElement,
    },
};
use eyre::{bail, OptionExt, Result};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    vector_id::VectorId,
};
use itertools::{izip, Itertools};
use static_assertions::const_assert;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    sync::Arc,
    vec,
};
use tracing::instrument;

mod distance_fn;
pub use distance_fn::DistanceFn;

/// The number of rotations at which to switch from binary tree to round-robin minimum algorthims.
const MIN_ROUND_ROBIN_SIZE: usize = 1;
const_assert!(MIN_ROUND_ROBIN_SIZE >= 1);

/// Iris to be searcher or inserted into the store.
///
/// This is an iris reference along with cached preprocessed version, used for
/// efficient Galois ring MPC comparison.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct Aby3Query {
    /// Iris in the Shamir secret shared form over a Galois ring.
    pub iris: ArcIris,

    /// Preprocessed iris for faster evaluation of distances; see [Aby3Store::eval_distance].
    pub iris_proc: ArcIris,
}

impl Aby3Query {
    /// Creates a new query from a secret shared iris. The input iris is preprocessed for
    /// faster evaluation of distances; see [Aby3Store::eval_distance].
    pub fn new(iris_ref: &ArcIris) -> Self {
        let iris = iris_ref.clone();

        let mut preprocessed = (**iris_ref).clone();
        preprocessed.code.preprocess_iris_code_query_share();
        preprocessed.mask.preprocess_mask_code_query_share();
        let iris_proc = Arc::new(preprocessed);

        Self { iris, iris_proc }
    }

    pub fn new_from_raw(iris: GaloisRingSharedIris) -> Self {
        let iris = Arc::new(iris);
        Self::new(&iris)
    }

    pub fn from_processed(
        code: &GaloisRingIrisCodeShare,
        mask: &GaloisRingTrimmedMaskCodeShare,
        code_proc: &GaloisRingIrisCodeShare,
        mask_proc: &GaloisRingTrimmedMaskCodeShare,
    ) -> Self {
        let iris = Arc::new(GaloisRingSharedIris {
            code: code.clone(),
            mask: mask.clone(),
        });
        let iris_proc = Arc::new(GaloisRingSharedIris {
            code: code_proc.clone(),
            mask: mask_proc.clone(),
        });
        Self { iris, iris_proc }
    }
}

pub type Aby3VectorRef = <Aby3Store as VectorStore>::VectorRef;
pub type Aby3DistanceRef = <Aby3Store as VectorStore>::DistanceRef;

pub type Aby3SharedIrises = SharedIrises<ArcIris>;
pub type Aby3SharedIrisesRef = SharedIrisesRef<ArcIris>;

/// Implementation of VectorStore based on the ABY3 framework (<https://eprint.iacr.org/2018/403.pdf>).
///
/// Note that all SMPC operations are performed in a single session.
#[derive(Debug)]
pub struct Aby3Store {
    /// Reference to the shared irises
    pub storage: Aby3SharedIrisesRef,

    /// Session for the SMPC operations
    pub session: Session,

    /// used to spawn cpu bound tasks on a thread pool
    pub workers: IrisPoolHandle,

    distance_fn: distance_fn::DistanceFn,
}

impl Aby3Store {
    pub fn new(
        storage: Aby3SharedIrisesRef,
        session: Session,
        workers: IrisPoolHandle,
        distance_fn: DistanceFn,
    ) -> Self {
        Self {
            storage,
            session,
            distance_fn,
            workers,
        }
    }

    /// Converts distances from u16 secret shares to u32 shares.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub(crate) async fn lift_distances(
        &mut self,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        let distances = batch_signed_lift_vec(&mut self.session, distances).await?;
        Ok(distances
            .chunks(2)
            .map(|dot_products| {
                DistanceShare::new(dot_products[0].clone(), dot_products[1].clone())
            })
            .collect::<Vec<_>>())
    }

    /// Converts u16 additive sharing (from trick_dot output) to u32 replicated sharing.
    async fn gr_to_lifted_distances(
        &mut self,
        ds_and_ts: Vec<RingElement<u16>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        let dist = galois_ring_to_rep3(&mut self.session, ds_and_ts).await?;
        self.lift_distances(dist).await
    }

    /// Computes the dot product of the iris codes and masks of the given pairs of irises.
    /// The input irises are given in the Shamir secret sharing scheme, while the output distances are additive replicated secret shares used in the ABY3 framework.
    ///
    /// Assumes that the first iris of each pair is preprocessed.
    /// This first iris is usually preprocessed when a related query is created, see [Aby3Query] for more details.
    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    pub async fn eval_pairwise_distances(
        &mut self,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<DistanceShare<u32>>> {
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        self.distance_fn.eval_pairwise_distances(self, pairs).await
    }

    /// Create a new `Aby3SharedIrises` storage using the specified points mapping.
    pub fn new_storage(points: Option<HashMap<VectorId, ArcIris>>) -> Aby3SharedIrises {
        SharedIrises::new(
            points.unwrap_or_default(),
            Arc::new(GaloisRingSharedIris::default_for_party(0)),
        )
    }

    pub async fn checksum(&self) -> u64 {
        self.storage.checksum().await
    }

    /// Obliviously swaps the elements in `list` at the given `indices` according to the `swap_bits`.
    /// If bit is 0, the elements are swapped, otherwise they are left unchanged.
    /// Note that unchanged elements of the list are propagated as secret-shares.
    pub async fn oblivious_swap_batch_plain_ids(
        &mut self,
        swap_bits: Vec<Share<Bit>>,
        list: &[(u32, Aby3DistanceRef)],
        indices: &[(usize, usize)],
    ) -> Result<Vec<(Share<u32>, Aby3DistanceRef)>> {
        if list.is_empty() {
            return Ok(vec![]);
        }

        conditionally_swap_distances_plain_ids(&mut self.session, swap_bits, list, indices).await
    }

    /// Obliviously compares pairs of distances in batch and returns a secret shared bit a < b for each pair.
    pub async fn oblivious_less_than_batch(
        &mut self,
        distances: &[(Aby3DistanceRef, Aby3DistanceRef)],
    ) -> Result<Vec<Share<Bit>>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        oblivious_cross_compare(&mut self.session, distances).await
    }

    /// Obliviously swaps the elements in `list` at the given `indices` according to the `swap_bits`.
    /// If bit is 0, the elements are swapped, otherwise they are left unchanged.
    pub async fn oblivious_swap_batch(
        &mut self,
        swap_bits: Vec<Share<Bit>>,
        list: &[(Share<u32>, Aby3DistanceRef)],
        indices: &[(usize, usize)],
    ) -> Result<Vec<(Share<u32>, Aby3DistanceRef)>> {
        if list.is_empty() {
            return Ok(vec![]);
        }

        conditionally_swap_distances(&mut self.session, swap_bits, list, indices).await
    }

    /// Obliviously computes the minimum distance of a given distance array.
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    pub async fn oblivious_min_distance(
        &mut self,
        distances: &[Aby3DistanceRef],
    ) -> Result<Aby3DistanceRef> {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        if distances.len() == 1 {
            return Ok(distances[0].clone());
        }
        let mut res = distances.to_vec();
        while res.len() > MIN_ROUND_ROBIN_SIZE {
            // if the length is odd, we save the last distance to add it back later
            let maybe_last_distance = if res.len() % 2 == 1 { res.pop() } else { None };
            // create pairs from the remaining distances
            let pairs: Vec<(_, _)> = res.into_iter().tuples().collect_vec();
            // compute minimums of pairs
            res = min_of_pair_batch(&mut self.session, &pairs).await?;
            // if we saved a last distance, we need to add it back
            if let Some(last_distance) = maybe_last_distance {
                res.push(last_distance.clone());
            }
        }
        min_round_robin_batch(&mut self.session, &res, res.len())
            .await?
            .pop()
            .ok_or_eyre("Should not be here: distances are empty")
    }

    /// Obliviously computes the minimum distance and the corresponding vector id of a given array of pairs (id, distance).
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    pub async fn oblivious_argmin_distance(
        &mut self,
        distances: &[(Aby3VectorRef, Aby3DistanceRef)],
    ) -> Result<(Aby3VectorRef, Aby3DistanceRef)> {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        if distances.len() == 1 {
            return Ok(distances[0].clone());
        }

        // Handle plain ids first
        let mut plain_res = distances
            .iter()
            .enumerate()
            .map(|(id, (_, distance))| (id as u32, distance.clone()))
            .collect_vec();
        let plain_maybe_last_distance = if plain_res.len() % 2 == 1 {
            plain_res.pop()
        } else {
            None
        };
        let mut dist_pairs = plain_res
            .iter()
            .tuples()
            .map(|((_, dist1), (_, dist2))| (dist1.clone(), dist2.clone()))
            .collect_vec();
        let mut control_bits =
            oblivious_cross_compare_lifted(&mut self.session, &dist_pairs).await?;
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
            let shared_id = Share::from_const(id, self.session.own_role());
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
                .map(|((_, dist1), (_, dist2))| (dist1.clone(), dist2.clone()))
                .collect_vec();
            // compute minimums of pairs
            control_bits = oblivious_cross_compare_lifted(&mut self.session, &dist_pairs).await?;
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
        let res = (distances[id as usize].0, dist);
        Ok(res)
    }

    /// Obliviously computes the minimum distance for each batch of given distances of the same size.
    /// The input `distances` is a 2D matrix with dimensions: (rotations, batch).
    /// `distances[r][i]` corresponds to the rth rotation of the ith item of the batch.
    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn oblivious_min_distance_batch(
        &mut self,
        distances: Vec<Vec<Aby3DistanceRef>>,
    ) -> Result<Vec<Aby3DistanceRef>> {
        if distances.is_empty() {
            eyre::bail!("Cannot compute minimum of empty list");
        }
        let len = distances[0].len();
        for (i, d) in distances.iter().enumerate() {
            if d.len() != len {
                eyre::bail!("All distance lists must have the same length. List at index {} has length {}, while the first list has length {}", i, d.len(), len);
            }
        }

        let mut res = distances.to_vec();
        while res.len() > MIN_ROUND_ROBIN_SIZE {
            // if the length is odd, we save the last distance to add it back later
            let maybe_last_distance = if res.len() % 2 == 1 { res.pop() } else { None };
            let pairs = res
                .into_iter()
                .tuples()
                .flat_map(|(a, b)| izip!(a, b).collect_vec())
                .collect_vec();
            // compute minimums of pairs
            let flattened_res = min_of_pair_batch(&mut self.session, &pairs).await?;
            res = flattened_res
                .into_iter()
                .chunks(len)
                .into_iter()
                .map(|chunk| chunk.collect())
                .collect_vec();
            // if we saved a last distance, we need to add it back
            if let Some(last_distance) = maybe_last_distance {
                res.push(last_distance.clone());
            }
        }
        let flattened_distances = res.iter().flatten().cloned().collect_vec();
        min_round_robin_batch(&mut self.session, &flattened_distances, res.len()).await
    }

    async fn compact_neighborhood_batch(
        &mut self,
        base_nodes: &[Aby3VectorRef],
        neighborhoods: &[Vec<Aby3VectorRef>],
        max_sizes: &[usize],
    ) -> Result<Vec<Vec<Aby3VectorRef>>> {
        if base_nodes.len() != neighborhoods.len() || base_nodes.len() != max_sizes.len() {
            bail!("Lists of base nodes, neighborhoods, and max sizes must have equal sizes");
        }

        let base_node_queries = self.vectors_as_queries(base_nodes.to_vec()).await;
        let query_vec_pairs = izip!(base_node_queries, neighborhoods.iter())
            .flat_map(|(q, nbhd)| nbhd.iter().map(move |nb| (q.clone(), *nb)))
            .collect_vec();
        let distances = self.eval_distance_pairs(&query_vec_pairs).await?;
        let id_distances = neighborhoods
            .iter()
            .flatten()
            .zip(distances)
            .map(|(vector_id, distance)| (vector_id.serial_id(), distance))
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
        let res_id_distances =
            apply_oblivious_swap_network(self, &id_distances, &batched_network).await?;

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
        let mut shuffled_shares_by_idx: BTreeMap<usize, Vec<_>> = BTreeMap::new();
        for (_len, v) in shares_by_length.into_iter() {
            let (idxs, nbhds): (Vec<_>, Vec<_>) = v.into_iter().unzip();

            let shuffled_nbhds =
                ampc_actor_utils::protocol::shuffle::random_shuffle_batch(&mut self.session, nbhds)
                    .await?;

            for (idx, shuffled_nbhd) in izip!(idxs, shuffled_nbhds) {
                shuffled_shares_by_idx.insert(idx, shuffled_nbhd);
            }
        }

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
                        let version = *id_versions.get(&(idx, serial_id)).ok_or_eyre(format!(
                            "Unexpected: found no record of reconstructed serial id: {}",
                            serial_id
                        ))?;
                        Ok(VectorId::new(serial_id, version))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(compacted_nbhds)
    }
}

impl VectorStore for Aby3Store {
    /// Arc ref to a query.
    type QueryRef = Aby3Query;
    /// Point ID of an inserted iris.
    type VectorRef = VectorId;
    /// Distance represented as a pair of u32 shares.
    type DistanceRef = DistanceShare<u32>;

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        self.storage
            .get_vectors_or_empty(&vectors)
            .await
            .iter()
            .map(Aby3Query::new)
            .collect_vec()
    }

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        let storage = self.storage.read().await;
        vectors.retain(|v| storage.contains(v));
        vectors
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        let mut d = self.eval_distance_batch(query, &[*vector]).await?;
        Ok(d.pop().unwrap())
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(queries = pairs.len(), batch_size = pairs.len()))]
    async fn eval_distance_pairs(
        &mut self,
        pairs: &[(Self::QueryRef, Self::VectorRef)],
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
        vectors: &[Self::VectorRef],
    ) -> Result<Vec<Self::DistanceRef>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        self.distance_fn
            .eval_distance_batch(self, query, vectors)
            .await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn get_argmin_distance(
        &mut self,
        distances: &[(Self::VectorRef, Self::DistanceRef)],
    ) -> Result<(Self::VectorRef, Self::DistanceRef)> {
        if distances.is_empty() {
            return Err(eyre::eyre!("Cannot get min of empty list"));
        }
        self.oblivious_argmin_distance(distances).await
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(lte_threshold_and_open(&mut self.session, &[distance.clone()]).await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        Ok(cross_compare(&mut self.session, &[(distance1.clone(), distance2.clone())]).await?[0])
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        cross_compare(&mut self.session, distances).await
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all, fields(batch_size = distances.len()))]
    async fn is_match_batch(&mut self, distances: &[Self::DistanceRef]) -> Result<Vec<bool>> {
        if distances.is_empty() {
            return Ok(vec![]);
        }
        lte_threshold_and_open(&mut self.session, distances).await
    }

    async fn compact_neighborhood(
        &mut self,
        base_node: Self::VectorRef,
        neighborhood: &[Self::VectorRef],
        max_size: usize,
    ) -> Result<Vec<Self::VectorRef>> {
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
        base_nodes: &[Self::VectorRef],
        neighborhoods: &[Vec<Self::VectorRef>],
        max_sizes: &[usize],
    ) -> Result<Vec<Vec<Self::VectorRef>>> {
        self.compact_neighborhood_batch(base_nodes, neighborhoods, max_sizes)
            .await
    }
}

impl VectorStoreMut for Aby3Store {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(&query.iris).await
    }

    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        Ok(self.storage.insert(*vector_ref, &query.iris).await)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::{
        execution::{hawk_main::scheduler::parallelize, session::SessionHandles},
        hawkers::{
            aby3::test_utils::{
                eval_vector_distance, get_owner_index, lazy_random_setup,
                setup_local_store_aby3_players, shared_random_setup,
            },
            plaintext_store::PlaintextStore,
        },
        hnsw::{GraphMem, HnswSearcher, SortedNeighborhood},
        network::NetworkType,
        protocol::shared_iris::GaloisRingSharedIris,
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::{izip, Itertools};
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_gr_hnsw() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let cleartext_database = IrisDB::new_random_rng(database_size, &mut rng).db;
        let shared_irises: Vec<_> = cleartext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        let stores = setup_local_store_aby3_players(NetworkType::Local).await?;

        let mut jobs = JoinSet::new();
        for store in stores.iter() {
            let player_index = get_owner_index(store).await?;
            let queries = (0..database_size)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect::<Vec<_>>();
            let mut rng = rng.clone();
            let store = store.clone();
            jobs.spawn(async move {
                let mut store = store.lock().await;
                let mut aby3_graph = GraphMem::new();
                let db = HnswSearcher::new_with_test_parameters();

                let mut inserted = vec![];
                // insert queries
                for query in queries.iter() {
                    let insertion_layer = db.gen_layer_rng(&mut rng).unwrap();
                    let inserted_vector = db
                        .insert::<_, SortedNeighborhood<_>>(
                            &mut *store,
                            &mut aby3_graph,
                            query,
                            insertion_layer,
                        )
                        .await
                        .unwrap();
                    inserted.push(inserted_vector)
                }
                tracing::debug!("FINISHED INSERTING");
                // Search for the same codes and find matches.
                let mut matching_results = vec![];
                for v in inserted.into_iter() {
                    let iris = store.storage.get_vector_or_empty(&v).await;
                    let query = Aby3Query::new(&iris);
                    let neighbors = db
                        .search::<_, SortedNeighborhood<_>>(&mut *store, &aby3_graph, &query, 1)
                        .await
                        .unwrap();
                    tracing::debug!("Finished checking query");
                    matching_results.push(db.is_match(&mut *store, &[neighbors]).await.unwrap())
                }
                matching_results
            });
        }
        let matching_results = jobs.join_all().await;
        for (party_id, party_results) in matching_results.iter().enumerate() {
            for (index, result) in party_results.iter().enumerate() {
                assert!(
                    *result,
                    "Failed at index {:?} for party {:?}",
                    index, party_id
                );
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_premade_hnsw() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 10;
        let network_t = NetworkType::Local;
        let (mut cleartext_data, secret_data) =
            lazy_random_setup(&mut rng, database_size, network_t.clone()).await?;

        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut vector_graph_stores =
            shared_random_setup(&mut rng, database_size, network_t).await?;

        for ((v_from_scratch, _), (premade_v, _)) in
            vector_graph_stores.iter().zip(secret_data.iter())
        {
            let v_from_scratch = v_from_scratch.lock().await;
            let premade_v = premade_v.lock().await;
            assert_eq!(
                v_from_scratch.storage.read().await.get_points(),
                premade_v.storage.read().await.get_points()
            );
        }
        let hawk_searcher = HnswSearcher::new_with_test_parameters();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let query = cleartext_data
                .0
                .storage
                .get_vector(&vector_id)
                .unwrap()
                .clone();
            let cleartext_neighbors = hawk_searcher
                .search::<_, SortedNeighborhood<_>>(
                    &mut cleartext_data.0,
                    &cleartext_data.1,
                    &query,
                    1,
                )
                .await?;
            assert!(
                hawk_searcher
                    .is_match(&mut cleartext_data.0, &[cleartext_neighbors])
                    .await?,
            );

            let mut jobs = JoinSet::new();
            for (v, g) in vector_graph_stores.iter_mut() {
                let hawk_searcher = hawk_searcher.clone();
                let v_lock = v.lock().await;
                let g = g.clone();
                let q = v_lock.storage.get_vector_or_empty(&vector_id).await;
                let q = Aby3Query::new(&q);
                let v = v.clone();
                jobs.spawn(async move {
                    let mut v_lock = v.lock().await;
                    let secret_neighbors: SortedNeighborhood<_> =
                        hawk_searcher.search(&mut *v_lock, &g, &q, 1).await.unwrap();

                    hawk_searcher
                        .is_match(&mut *v_lock, &[secret_neighbors])
                        .await
                });
            }
            let scratch_results = jobs.join_all().await;

            let mut jobs = JoinSet::new();
            for (v, g) in secret_data.iter() {
                let hawk_searcher = hawk_searcher.clone();
                let v = v.clone();
                let g = g.clone();
                jobs.spawn(async move {
                    let mut v_lock = v.lock().await;
                    let iris = v_lock.storage.get_vector_or_empty(&vector_id).await;
                    let query = Aby3Query::new(&iris);
                    let secret_neighbors: SortedNeighborhood<_> = hawk_searcher
                        .search(&mut *v_lock, &g, &query, 1)
                        .await
                        .unwrap();

                    hawk_searcher
                        .is_match(&mut *v_lock, &[secret_neighbors])
                        .await
                });
            }
            let premade_results = jobs.join_all().await;

            for (premade_res, scratch_res) in izip!(scratch_results, premade_results) {
                assert!(premade_res?);
                assert!(scratch_res?);
            }
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_dim = 4;
        let plaintext_database = IrisDB::new_random_rng(db_dim, &mut rng).db;
        let shared_irises: Vec<_> = plaintext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::new();
        let plaintext_preps: Vec<_> = (0..db_dim)
            .map(|id| Arc::new(plaintext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::new();
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // pairs of indices to compare
        let it1 = (0..db_dim).combinations(2);
        let it2 = (0..db_dim).combinations(2);

        let plaintext_queries: Vec<_> = plaintext_database.into_iter().map(Arc::new).collect();

        let mut plain_results = HashMap::new();
        for comb1 in it1.clone() {
            for comb2 in it2.clone() {
                // compute distances in plaintext
                let dist1_plain = plaintext_store
                    .eval_distance(&plaintext_queries[comb1[0]], &plaintext_inserts[comb1[1]])
                    .await?;
                let dist2_plain = plaintext_store
                    .eval_distance(&plaintext_queries[comb2[0]], &plaintext_inserts[comb2[1]])
                    .await?;
                let bit = plaintext_store
                    .less_than(&dist1_plain, &dist2_plain)
                    .await?;
                plain_results.insert((comb1.clone(), comb2.clone()), bit);
            }
        }

        let mut aby3_inserts = vec![];
        for store in local_stores.iter_mut() {
            let player_index = get_owner_index(store).await?;
            let player_preps: Vec<_> = (0..db_dim)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect();
            let mut player_inserts = vec![];
            let mut store_lock = store.lock().await;
            for p in player_preps.iter() {
                player_inserts.push(store_lock.storage.append(&p.iris).await);
            }
            aby3_inserts.push(player_inserts);
        }

        for comb1 in it1 {
            for comb2 in it2.clone() {
                let mut jobs = JoinSet::new();
                for store in local_stores.iter() {
                    let player_index = get_owner_index(store).await?;
                    let player_inserts = aby3_inserts[player_index].clone();
                    let store = store.clone();
                    let index10 = comb1[0];
                    let index11 = comb1[1];
                    let index20 = comb2[0];
                    let index21 = comb2[1];
                    jobs.spawn(async move {
                        let mut store = store.lock().await;
                        let dist1_aby3 = eval_vector_distance(
                            &mut store,
                            &player_inserts[index10],
                            &player_inserts[index11],
                        )
                        .await?;
                        let dist2_aby3 = eval_vector_distance(
                            &mut store,
                            &player_inserts[index20],
                            &player_inserts[index21],
                        )
                        .await?;
                        store.less_than(&dist1_aby3, &dist2_aby3).await
                    });
                }
                let res = jobs.join_all().await;
                for bit in res {
                    assert_eq!(
                        bit?,
                        plain_results[&(comb1.clone(), comb2.clone())],
                        "Failed at combo: {:?}, {:?}",
                        comb1,
                        comb2
                    )
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_oblivious_swap() -> Result<()> {
        let list_len = 6_u32;
        let plain_list = (0..list_len)
            .map(|i| (VectorId::from_0_index(i), (i, i)))
            .collect_vec();
        let swap_bits_for_plain = vec![true, false];
        let indices_for_plain = vec![(0, 1), (4, 5)];
        let swap_bits_for_secret = vec![true, false, false];
        let indices_for_secret = vec![(1, 2), (0, 4), (3, 5)];

        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let store = store.clone();
            let swap_bits_for_plain = swap_bits_for_plain.clone();
            let swap_bits_for_secret = swap_bits_for_secret.clone();
            let plain_list = plain_list.clone();
            let indices_for_plain = indices_for_plain.clone();
            let indices_for_secret = indices_for_secret.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let role = store_lock.session.own_role();
                let swap_bits1 = swap_bits_for_plain
                    .iter()
                    .map(|b| Share::from_const(Bit::new(*b), role))
                    .collect_vec();
                let swap_bits2 = swap_bits_for_secret
                    .iter()
                    .map(|b| Share::from_const(Bit::new(*b), role))
                    .collect_vec();
                let list = plain_list
                    .iter()
                    .map(|(v, d)| {
                        (
                            v.index(),
                            DistanceShare::new(
                                Share::from_const(d.0, role),
                                Share::from_const(d.1, role),
                            ),
                        )
                    })
                    .collect_vec();
                let tmp_list = store_lock
                    .oblivious_swap_batch_plain_ids(swap_bits1, &list, &indices_for_plain)
                    .await?;
                store_lock
                    .oblivious_swap_batch(swap_bits2, &tmp_list, &indices_for_secret)
                    .await
            });
        }
        let res = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let mut expected_list = plain_list.clone();
        expected_list.swap(4, 5);
        expected_list.swap(0, 4);
        expected_list.swap(3, 5);

        for (i, exp) in expected_list.iter().enumerate() {
            let id = (res[0][i].clone().0 + &res[1][i].0 + &res[2][i].0)
                .get_a()
                .convert();
            assert_eq!(id, exp.0.index());

            let distance = {
                let code_dot =
                    (res[0][i].clone().1.code_dot + &res[1][i].1.code_dot + &res[2][i].1.code_dot)
                        .get_a()
                        .convert();
                let mask_dot =
                    (res[0][i].clone().1.mask_dot + &res[1][i].1.mask_dot + &res[2][i].1.mask_dot)
                        .get_a()
                        .convert();
                (code_dot, mask_dot)
            };
            assert_eq!(distance, exp.1);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_oblivious_min() -> Result<()> {
        let list_len = 6_u32;
        let mut plain_list = (0..list_len).map(|i| (i, 1)).collect_vec();
        // place the smallest distance at index 3
        plain_list.swap(5, 3);

        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let store = store.clone();
            let plain_list = plain_list.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let role = store_lock.session.own_role();
                let list = plain_list
                    .iter()
                    .map(|(code_dist, mask_dist)| {
                        DistanceShare::new(
                            Share::from_const(*code_dist, role),
                            Share::from_const(*mask_dist, role),
                        )
                    })
                    .collect_vec();
                store_lock.oblivious_min_distance(&list).await
            });
        }
        let res = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let expected = plain_list
            .into_iter()
            .min_by(|a, b| (b.0 * a.1).cmp(&(a.0 * b.1)))
            .unwrap();

        let distance = {
            let code_dot = (res[0].clone().code_dot + &res[1].code_dot + &res[2].code_dot)
                .get_a()
                .convert();
            let mask_dot = (res[0].clone().mask_dot + &res[1].mask_dot + &res[2].mask_dot)
                .get_a()
                .convert();
            (code_dot, mask_dot)
        };
        assert_eq!(distance, expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_oblivious_argmin() -> Result<()> {
        let list_len = 6_u32;
        let mut plain_list = (0..list_len).map(|i| (i, (i, 1))).collect_vec();
        // place the smallest distance at index 3
        plain_list.swap(5, 3);

        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let store = store.clone();
            let plain_list = plain_list.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let role = store_lock.session.own_role();
                let list = plain_list
                    .iter()
                    .map(|(id, (code_dist, mask_dist))| {
                        (
                            VectorId::from_serial_id(*id),
                            DistanceShare::new(
                                Share::from_const(*code_dist, role),
                                Share::from_const(*mask_dist, role),
                            ),
                        )
                    })
                    .collect_vec();
                store_lock.get_argmin_distance(&list).await
            });
        }
        let res = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let expected = plain_list
            .into_iter()
            .min_by(|(_, a), (_, b)| (b.0 * a.1).cmp(&(a.0 * b.1)))
            .unwrap();

        let distance = {
            let id = res[0].0;
            assert_eq!(id, res[1].0);
            assert_eq!(id, res[2].0);

            let (id, dist) = res
                .into_iter()
                .reduce(|(acc_id, acc_d), (_, a_d)| {
                    let code_dist = acc_d.code_dot + &a_d.code_dot;
                    let mask_dist = acc_d.mask_dot + &a_d.mask_dot;
                    (acc_id, DistanceShare::new(code_dist, mask_dist))
                })
                .unwrap();
            let code_dot = dist.code_dot.get_a().convert();
            let mask_dot = dist.mask_dot.get_a().convert();

            (id.serial_id(), (code_dot, mask_dot))
        };
        assert_eq!(distance, expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_oblivious_min_batch() -> Result<()> {
        let list_len = 6_u32;
        let num_lists = 3;
        // create 3 lists of length 6
        // [[(1,1), (2,1), (3,1), (4,1), (6,1), (5,1)],
        // [(7,1), (8,1), (9,1), (12,1), (10,1), (11,1)],
        // [(13,1), (14,1), (18,1), (15,1), (16,1), (17,1)]]
        let mut flat_list = (1..=(list_len * num_lists)).map(|i| (i, 1)).collect_vec();
        flat_list.swap(5, 4);
        flat_list.swap(11, 9);
        flat_list.swap(17, 14);
        // [(1,1), (7,1), (13,1)],
        // [(2,1), (8,1), (14,1)],
        // [(3,1), (9,1), (18,1)],
        // [(4,1), (12,1), (15,1)],
        // [(6,1), (10,1), (16,1)],
        // [(5,1), (11,1), (17,1)]
        let mut plain_list = Vec::with_capacity(list_len as usize);
        for i in 0..list_len {
            let mut slice = Vec::with_capacity(num_lists as usize);
            for j in 0..num_lists {
                slice.push(flat_list[(i + list_len * j) as usize]);
            }
            plain_list.push(slice);
        }

        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let store = store.clone();
            let plain_list = plain_list.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let role = store_lock.session.own_role();
                let list = plain_list
                    .iter()
                    .map(|sub_list| {
                        sub_list
                            .iter()
                            .map(|(code_dist, mask_dist)| {
                                DistanceShare::new(
                                    Share::from_const(*code_dist, role),
                                    Share::from_const(*mask_dist, role),
                                )
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                store_lock.oblivious_min_distance_batch(list).await
            });
        }
        let res = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let expected = flat_list
            .chunks_exact(list_len as usize)
            .map(|sublist| {
                sublist
                    .iter()
                    .min_by(|a, b| (b.0 * a.1).cmp(&(a.0 * b.1)))
                    .unwrap()
            })
            .collect_vec();

        for (i, exp) in expected.into_iter().enumerate() {
            let distance = {
                let code_dot =
                    (res[0][i].clone().code_dot + &res[1][i].code_dot + &res[2][i].code_dot)
                        .get_a()
                        .convert();
                let mask_dot =
                    (res[0][i].clone().mask_dot + &res[1][i].mask_dot + &res[2][i].mask_dot)
                        .get_a()
                        .convert();
                (code_dot, mask_dot)
            };
            assert_eq!(distance, *exp);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_aby3_store_plaintext_batch() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let db_size = 10;
        let plaintext_database = IrisDB::new_random_rng(db_size, &mut rng).db;
        let shared_irises: Vec<_> = plaintext_database
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();
        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        // Now do the work for the plaintext store
        let mut plaintext_store = PlaintextStore::new();
        let plaintext_preps: Vec<_> = (0..db_size)
            .map(|id| Arc::new(plaintext_database[id].clone()))
            .collect();
        let mut plaintext_inserts = Vec::with_capacity(db_size);
        for p in plaintext_preps.iter() {
            plaintext_inserts.push(plaintext_store.insert(p).await);
        }

        // compute distances in plaintext
        let dist1_plain = plaintext_store
            .eval_distance_batch(&plaintext_preps[0], &plaintext_inserts)
            .await?;
        let dist2_plain = plaintext_store
            .eval_distance_batch(&plaintext_preps[1], &plaintext_inserts)
            .await?;
        let dist_plain = dist1_plain
            .into_iter()
            .zip(dist2_plain.into_iter())
            .collect::<Vec<_>>();
        let bits_plain = plaintext_store.less_than_batch(&dist_plain).await?;

        let mut aby3_inserts = vec![];
        let mut queries = vec![];
        for store in local_stores.iter_mut() {
            let player_index = get_owner_index(store).await?;
            let player_preps: Vec<_> = (0..db_size)
                .map(|id| Aby3Query::new_from_raw(shared_irises[id][player_index].clone()))
                .collect();
            queries.push(player_preps.clone());
            let mut player_inserts = vec![];
            let mut store_lock = store.lock().await;
            for p in player_preps.iter() {
                player_inserts.push(store_lock.insert(p).await);
            }
            aby3_inserts.push(player_inserts);
        }

        let mut jobs = JoinSet::new();
        for store in local_stores.iter() {
            let player_index = get_owner_index(store).await?;
            let player_inserts = aby3_inserts[player_index].clone();
            let player_preps = queries[player_index].clone();
            let store = store.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let dist1_aby3 = store_lock
                    .eval_distance_batch(&player_preps[0], &player_inserts)
                    .await?;
                let dist2_aby3 = store_lock
                    .eval_distance_batch(&player_preps[1], &player_inserts)
                    .await?;
                let dist_aby3 = dist1_aby3
                    .into_iter()
                    .zip(dist2_aby3.into_iter())
                    .collect::<Vec<_>>();
                store_lock.less_than_batch(&dist_aby3).await
            });
        }
        let bits_aby3 = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        assert_eq!(bits_aby3[0], bits_aby3[1]);
        assert_eq!(bits_aby3[0], bits_aby3[2]);
        assert_eq!(bits_aby3[0], bits_plain);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_scratch_hnsw() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut vectors_and_graphs =
            shared_random_setup(&mut rng, database_size, NetworkType::Local)
                .await
                .unwrap();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let mut jobs = JoinSet::new();
            for (store, graph) in vectors_and_graphs.iter_mut() {
                let graph = graph.clone();
                let searcher = searcher.clone();
                let q = store
                    .lock()
                    .await
                    .storage
                    .get_vector_or_empty(&vector_id)
                    .await;
                let q = Aby3Query::new(&q);
                let store = store.clone();
                jobs.spawn(async move {
                    let mut store = store.lock().await;
                    let secret_neighbors: SortedNeighborhood<_> =
                        searcher.search(&mut *store, &graph, &q, 1).await.unwrap();
                    searcher
                        .is_match(&mut *store, &[secret_neighbors])
                        .await
                        .unwrap()
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.into_iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_gr_non_existent_vectors() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let vectors_and_graphs = shared_random_setup(&mut rng, database_size, NetworkType::Local)
            .await
            .unwrap();

        let mut tasks = vec![];
        for (store, _graph) in vectors_and_graphs {
            let mut store = store.lock_owned().await;
            tasks.push(async move {
                let none = VectorId::from_0_index(999);
                let a = VectorId::from_0_index(0);
                let b = VectorId::from_0_index(1);

                let queries = store.vectors_as_queries(vec![a, b]).await;
                let vectors = vec![a, b, none];
                let n_vecs = vectors.len();

                let distances = {
                    let mut dist_a = store
                        .eval_distance_batch(&queries[0], &vectors)
                        .await
                        .unwrap();
                    let dist_b = store
                        .eval_distance_batch(&queries[1], &vectors)
                        .await
                        .unwrap();
                    dist_a.extend(dist_b);
                    dist_a
                };

                let is_match = store.is_match_batch(&distances).await.unwrap();
                assert_eq!(
                    is_match,
                    [vec![true, false, false], vec![false, true, false]].concat(),
                    "Vectors should match with themselves and not with the others"
                );

                let distances_to_none = vec![distances[2].clone(), distances[2 + n_vecs].clone()];
                let pairs = distances_to_none
                    .into_iter()
                    .cartesian_product(distances)
                    .collect_vec();
                let less_than = store.less_than_batch(&pairs).await.unwrap();

                assert_eq!(
                    less_than,
                    vec![false; pairs.len()],
                    "Nothing is less than a distance to a non-existent vector"
                );

                Ok(())
            });
        }
        parallelize(tasks.into_iter()).await.unwrap();
    }
}
