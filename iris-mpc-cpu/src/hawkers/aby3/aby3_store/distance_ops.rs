use std::{cmp::Ordering, fmt::Debug};

use ampc_actor_utils::{
    execution::session::Session,
    network::mpc::NetworkInt,
    protocol::{
        binary::{bit_inject, extract_msb_batch, open_bin},
        fhd_ops::{cross_mul, fhd_greater_than_threshold},
        nhd_ops::{
            nhd_comparison_nmr, nhd_cross_mul, nhd_greater_than_threshold, nhd_lift_distances,
            nhd_plaintext_is_match,
        },
        ops::{batch_signed_lift_vec, conditionally_select_distance},
        shuffle::random_shuffle_batch,
    },
};
use ampc_secret_sharing::{
    shares::{
        bit::Bit, vecshare_bittranspose::Transpose64, DistanceShare, Ring48, RingRandFillable,
        VecShare,
    },
    IntRing2k, Share,
};
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use rand_distr::{Distribution, Standard};

use iris_mpc_common::iris_db::iris::Threshold;

use crate::{
    hawkers::aby3::aby3_store::DistanceFn,
    protocol::{
        min_round_robin::{build_round_robin_pairs, select_round_robin_min},
        ops::{DistancePair, IdDistance},
    },
};

use crate::execution::hawk_main::HAWK_MIN_DIST_ROTATIONS;
/// Trait abstracting distance-type-specific MPC operations.
///
/// Parameterized by marker types (not ring types), so multiple protocols
/// can share the same ring type.
///
/// Implementations:
/// - `FhdOps`: Fractional Hamming Distance (FHD) using 32-bit arithmetic
/// - `NhdOps`: Normalized Hamming Distance (NHD) using 48-bit arithmetic
#[allow(async_fn_in_trait)]
pub trait DistanceOps: Send + Sync + Debug + 'static {
    type Ring: IntRing2k
        + NetworkInt
        + RingRandFillable
        + From<u32>
        + Debug
        + std::hash::Hash
        + Eq
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>
        + Send
        + Sync;

    /// Lifts u16 distance shares to Ring-typed distance shares.
    async fn lift_distances(
        session: &mut Session,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<Self::Ring>>>;

    /// Cross-multiplies distance pairs to produce a difference value.
    /// FHD: simple cross-multiply; NHD: polynomial NMR cross-multiply.
    async fn cross_mul(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Self::Ring>>>;

    /// Compares pairs of distances, returning secret-shared bits.
    async fn oblivious_cross_compare(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Bit>>>
    where
        VecShare<Self::Ring>: Transpose64,
        Standard: Distribution<Self::Ring>,
    {
        let diff = Self::cross_mul(session, distances).await?;
        extract_msb_batch(session, &diff).await
    }

    /// Compares pairs of distances and opens the result (d1 < d2 for each pair).
    async fn cross_compare(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<bool>>
    where
        VecShare<Self::Ring>: Transpose64,
        Standard: Distribution<Self::Ring>,
    {
        let bits = Self::oblivious_cross_compare(session, distances).await?;
        let opened_b = open_bin(session, &bits).await?;
        opened_b.into_iter().map(|x| Ok(x.convert())).collect()
    }

    /// Compares pairs of distances, returning lifted Ring-typed shares.
    async fn oblivious_cross_compare_lifted(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Self::Ring>>>
    where
        VecShare<Self::Ring>: Transpose64,
        Standard: Distribution<Self::Ring>,
    {
        let bits = Self::oblivious_cross_compare(session, distances).await?;
        Ok(bit_inject::<Self::Ring>(session, VecShare { shares: bits })
            .await?
            .inner())
    }

    /// Computes the minimum of each pair of distances.
    async fn min_of_pair_batch(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<DistanceShare<Self::Ring>>>
    where
        VecShare<Self::Ring>: Transpose64,
        Standard: Distribution<Self::Ring>,
    {
        let bits = Self::oblivious_cross_compare_lifted(session, distances).await?;
        conditionally_select_distance(session, distances, bits.as_slice()).await
    }

    /// Given a flattened array of distance shares arranged in batches,
    /// this function computes the minimum distance share within each batch via the round-robin method.
    ///
    /// If `d[i][j]` is the ith distance share of the jth batch and `num_batches` is the number of input batches,
    /// then the input distance shares are arranged as follows:
    /// `[
    ///     d[0][0],            d[0][1],            ..., d[0][num_batches-1], // first elements of each batch
    ///     d[1][0],            d[1][1],            ..., d[1][num_batches-1], // second elements of each batch
    ///     ...,
    ///     d[batch_size-1][0], d[batch_size-1][1], ..., d[batch_size-1][num_batches-1] // last elements of each batch
    /// ]`
    ///
    /// The round-robin method computes all pairwise "less-than" relations within each batch,
    /// and puts them into a comparison table. For example, for a batch size of 4, the comparison table looks like
    ///
    ///    | d0 | d1 | d2 | d3 |
    /// ------------------------
    /// d0 | 1  | b01| b02| b03|
    /// d1 | b10| 1  | b12| b13|
    /// d2 | b20| b21| 1  | b23|
    /// d3 | b30| b31| b32| 1  |
    ///
    /// where `bij` is the bit corresponding to `di < dj` if `i < j`, and `bij` is the bit `di <= dj` if `i > j`.
    /// The latter bits are in fact negations of the former bits, i.e., if `i > j`, `bij = !(di > dj) = !bji`,
    /// that turns the comparison table into
    ///
    ///    | d0 | d1 | d2 | d3 |
    /// ------------------------
    /// d0 | 1  | b01| b02| b03|
    /// d1 |!b01| 1  | b12| b13|
    /// d2 |!b02|!b12| 1  | b23|
    /// d3 |!b03|!b13|!b23| 1  |
    ///
    /// The minimum distance in each batch can then be identified by ANDing each row of the comparison table.
    /// If the ith distance is the minimum in its batch, then all bits in the ith row are 1, and the AND of the row is 1.
    /// If there are two or more minimum distances in the batch, then the AND of the one with the greatest index will be 1.
    /// To see that, take such a minimum distance `dj`. For any `di = dj`, `i < j`, which means that `bij = 0` and `bji = 1`.
    /// Thus, only one row of the above table will have all 1s and the AND of that row will indicate the minimum distance in the batch.
    async fn min_round_robin_batch(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        batch_size: usize,
    ) -> Result<Vec<DistanceShare<Self::Ring>>>
    where
        VecShare<Self::Ring>: Transpose64,
        Standard: Distribution<Self::Ring>,
    {
        if distances.is_empty() {
            eyre::bail!("Expected at least one distance share");
        }
        if !distances.len().is_multiple_of(batch_size) {
            eyre::bail!("Distances length must be a multiple of batch size");
        }
        if batch_size < 2 {
            return Ok(distances.to_vec());
        }
        let num_batches = distances.len() / batch_size;
        let pairs = build_round_robin_pairs(distances, batch_size, num_batches);
        let comparison_bits = Self::oblivious_cross_compare(session, &pairs).await?;
        select_round_robin_min(session, distances, batch_size, num_batches, comparison_bits).await
    }

    /// Computes secret-shared bits indicating whether each distance exceeds the threshold.
    async fn greater_than_threshold(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        threshold: Threshold,
    ) -> Result<Vec<Share<Bit>>>;

    /// Checks if distances are less than or equal to the given threshold.
    async fn lte_and_open(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        threshold: Threshold,
    ) -> Result<Vec<bool>> {
        let gt_bits = Self::greater_than_threshold(session, distances, threshold).await?;
        let opened = open_bin(session, &gt_bits).await?;
        Ok(opened.into_iter().map(|b| !bool::from(b)).collect())
    }

    /// Converts an opened ring value to a usize index.
    fn to_usize(value: Self::Ring) -> usize;

    /// Plaintext comparison: returns true if d1 < d2.
    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool;

    /// Plaintext check: returns true if the distance represents a match (at or below threshold).
    fn plaintext_is_match(d: &(u16, u16)) -> bool;

    /// Plaintext ordering of two distances.
    fn plaintext_ordering(d1: &(u16, u16), d2: &(u16, u16)) -> Ordering;

    /// Plaintext distance computation between two iris codes via distance computation strategy f,
    /// returning a fractional Hamming distance pair (dot product of iris codes, dot product of mask codes).
    fn plaintext_distance(a: &IrisCode, b: &IrisCode, f: DistanceFn) -> (u16, u16);

    /// Shuffles batched (id, distance) pairs using the 3-party shuffle protocol.
    async fn shuffle_batch(
        session: &mut Session,
        distances: Vec<Vec<IdDistance<Self::Ring>>>,
    ) -> Result<Vec<Vec<IdDistance<Self::Ring>>>>
    where
        Standard: Distribution<Self::Ring>,
    {
        random_shuffle_batch(session, distances).await
    }
}

/// Fractional Hamming Distance operations using 32-bit arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct FhdOps;

impl DistanceOps for FhdOps {
    type Ring = u32;

    async fn lift_distances(
        session: &mut Session,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<Self::Ring>>> {
        let distances = batch_signed_lift_vec(session, distances).await?;
        Ok(distances
            .chunks(2)
            .map(|dot_products| DistanceShare::new(dot_products[0], dot_products[1]))
            .collect())
    }

    async fn cross_mul(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Self::Ring>>> {
        cross_mul(session, distances).await
    }

    async fn greater_than_threshold(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        threshold: Threshold,
    ) -> Result<Vec<Share<Bit>>> {
        fhd_greater_than_threshold(session, distances, threshold.ratio()).await
    }

    fn to_usize(value: Self::Ring) -> usize {
        value as usize
    }

    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool {
        let (a, b) = *d1; // a/b
        let (c, d) = *d2; // c/d
        (a as u32) * (d as u32) < (b as u32) * (c as u32)
    }

    fn plaintext_is_match(d: &(u16, u16)) -> bool {
        let (a, b) = *d;
        (a as f64) < (b as f64) * Threshold::Match.ratio()
    }

    fn plaintext_ordering(d1: &(u16, u16), d2: &(u16, u16)) -> Ordering {
        let (a, b) = *d1; // a/b
        let (c, d) = *d2; // c/d
        ((a as u32) * (d as u32)).cmp(&((b as u32) * (c as u32)))
    }

    fn plaintext_distance(a: &IrisCode, b: &IrisCode, f: DistanceFn) -> (u16, u16) {
        match f {
            DistanceFn::Simple => a.get_distance_fraction(b),
            DistanceFn::MinRotation => {
                a.get_min_fhd_distance_fraction_rotation_aware::<HAWK_MIN_DIST_ROTATIONS>(b)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NhdOps;

impl DistanceOps for NhdOps {
    type Ring = Ring48;

    async fn lift_distances(
        session: &mut Session,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<Self::Ring>>> {
        nhd_lift_distances(session, distances).await
    }

    async fn cross_mul(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Self::Ring>>> {
        nhd_cross_mul(session, distances).await
    }

    async fn greater_than_threshold(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        threshold: Threshold,
    ) -> Result<Vec<Share<Bit>>> {
        nhd_greater_than_threshold(session, distances, threshold.ratio()).await
    }

    fn to_usize(value: Self::Ring) -> usize {
        value.0 as usize
    }

    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool {
        let nmr1 = nhd_comparison_nmr(d1.0, d1.1);
        let nmr2 = nhd_comparison_nmr(d2.0, d2.1);
        nmr1 * (d2.1 as i64) < nmr2 * (d1.1 as i64)
    }

    fn plaintext_is_match(d: &(u16, u16)) -> bool {
        nhd_plaintext_is_match(d.0, d.1, Threshold::Match.ratio())
    }

    fn plaintext_ordering(d1: &(u16, u16), d2: &(u16, u16)) -> Ordering {
        let nmr1 = nhd_comparison_nmr(d1.0, d1.1);
        let nmr2 = nhd_comparison_nmr(d2.0, d2.1);
        (nmr1 * (d2.1 as i64)).cmp(&(nmr2 * (d1.1 as i64)))
    }

    fn plaintext_distance(a: &IrisCode, b: &IrisCode, f: DistanceFn) -> (u16, u16) {
        match f {
            DistanceFn::Simple => a.get_distance_fraction(b),
            DistanceFn::MinRotation => {
                a.get_min_nhd_distance_fraction_rotation_aware::<HAWK_MIN_DIST_ROTATIONS>(b)
            }
        }
    }
}
