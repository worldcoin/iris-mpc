use std::{cmp::Ordering, fmt::Debug};

use ampc_actor_utils::{
    constants::MATCH_THRESHOLD_RATIO,
    execution::session::Session,
    network::value::NetworkInt,
    protocol::{
        ops::{
            batch_signed_lift_vec, min_of_pair_batch, oblivious_cross_compare,
            oblivious_cross_compare_lifted,
        },
        shuffle::random_shuffle_batch,
    },
};
use ampc_secret_sharing::{
    shares::{bit::Bit, DistanceShare, Ring48, RingRandFillable},
    IntRing2k, Share,
};
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use rand_distr::{Distribution, Standard};

use crate::{
    hawkers::aby3::aby3_store::DistanceFn,
    protocol::{
        nhd_ops::{
            nhd_compare_nmr, nhd_cross_compare, nhd_lift_distances, nhd_lte_threshold_and_open,
            nhd_min_of_pair_batch, nhd_min_round_robin_batch, nhd_oblivious_cross_compare,
            nhd_oblivious_cross_compare_lifted, nhd_plaintext_is_match,
        },
        ops::{
            cross_compare, lte_threshold_and_open, min_round_robin_batch, DistancePair, IdDistance,
        },
    },
};

use crate::execution::hawk_main::HAWK_MIN_ROTATIONS;
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

    /// Compares pairs of distances and opens the result (d2 < d1 for each pair).
    async fn cross_compare(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<bool>>;

    /// Compares pairs of distances, returning secret-shared bits.
    async fn oblivious_cross_compare(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Bit>>>;

    /// Compares pairs of distances, returning lifted Ring-typed shares.
    async fn oblivious_cross_compare_lifted(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<Share<Self::Ring>>>;

    /// Computes the minimum of each pair of distances.
    async fn min_of_pair_batch(
        session: &mut Session,
        distances: &[DistancePair<Self::Ring>],
    ) -> Result<Vec<DistanceShare<Self::Ring>>>;

    /// Computes the minimum distance per batch via round-robin comparison.
    async fn min_round_robin_batch(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
        batch_size: usize,
    ) -> Result<Vec<DistanceShare<Self::Ring>>>;

    /// Checks if distances are less than or equal to a threshold (for match detection).
    async fn lte_threshold_and_open(
        session: &mut Session,
        distances: &[DistanceShare<Self::Ring>],
    ) -> Result<Vec<bool>>;

    /// Converts an opened ring value to a usize index.
    fn to_usize(value: Self::Ring) -> usize;

    /// Plaintext comparison: returns true if d1 < d2.
    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool;

    /// Plaintext check: returns true if the distance represents a match (below threshold).
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
    ) -> Result<Vec<DistanceShare<u32>>> {
        let distances = batch_signed_lift_vec(session, distances).await?;
        Ok(distances
            .chunks(2)
            .map(|dot_products| DistanceShare::new(dot_products[0], dot_products[1]))
            .collect())
    }

    async fn cross_compare(
        session: &mut Session,
        distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
    ) -> Result<Vec<bool>> {
        cross_compare(session, distances).await
    }

    async fn oblivious_cross_compare(
        session: &mut Session,
        distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
    ) -> Result<Vec<Share<Bit>>> {
        oblivious_cross_compare(session, distances).await
    }

    async fn oblivious_cross_compare_lifted(
        session: &mut Session,
        distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
    ) -> Result<Vec<Share<u32>>> {
        oblivious_cross_compare_lifted(session, distances).await
    }

    async fn min_of_pair_batch(
        session: &mut Session,
        distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
    ) -> Result<Vec<DistanceShare<u32>>> {
        min_of_pair_batch(session, distances).await
    }

    async fn min_round_robin_batch(
        session: &mut Session,
        distances: &[DistanceShare<u32>],
        batch_size: usize,
    ) -> Result<Vec<DistanceShare<u32>>> {
        min_round_robin_batch(session, distances, batch_size).await
    }

    async fn lte_threshold_and_open(
        session: &mut Session,
        distances: &[DistanceShare<u32>],
    ) -> Result<Vec<bool>> {
        lte_threshold_and_open(session, distances).await
    }

    fn to_usize(value: u32) -> usize {
        value as usize
    }

    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool {
        let (a, b) = *d1; // a/b
        let (c, d) = *d2; // c/d
        (a as u32) * (d as u32) < (b as u32) * (c as u32)
    }

    fn plaintext_is_match(d: &(u16, u16)) -> bool {
        let (a, b) = *d;
        (a as f64) < (b as f64) * MATCH_THRESHOLD_RATIO
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
                a.get_min_fhd_distance_fraction_rotation_aware::<HAWK_MIN_ROTATIONS>(b)
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct NhdOps;

impl DistanceOps for NhdOps {
    type Ring = Ring48;

    async fn lift_distances(
        session: &mut Session,
        distances: Vec<Share<u16>>,
    ) -> Result<Vec<DistanceShare<Ring48>>> {
        nhd_lift_distances(session, distances).await
    }

    async fn cross_compare(
        session: &mut Session,
        distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
    ) -> Result<Vec<bool>> {
        nhd_cross_compare(session, distances).await
    }

    async fn oblivious_cross_compare(
        session: &mut Session,
        distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
    ) -> Result<Vec<Share<Bit>>> {
        nhd_oblivious_cross_compare(session, distances).await
    }

    async fn oblivious_cross_compare_lifted(
        session: &mut Session,
        distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
    ) -> Result<Vec<Share<Ring48>>> {
        nhd_oblivious_cross_compare_lifted(session, distances).await
    }

    async fn min_of_pair_batch(
        session: &mut Session,
        distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
    ) -> Result<Vec<DistanceShare<Ring48>>> {
        nhd_min_of_pair_batch(session, distances).await
    }

    async fn min_round_robin_batch(
        session: &mut Session,
        distances: &[DistanceShare<Ring48>],
        batch_size: usize,
    ) -> Result<Vec<DistanceShare<Ring48>>> {
        nhd_min_round_robin_batch(session, distances, batch_size).await
    }

    async fn lte_threshold_and_open(
        session: &mut Session,
        distances: &[DistanceShare<Ring48>],
    ) -> Result<Vec<bool>> {
        nhd_lte_threshold_and_open(session, distances).await
    }

    fn to_usize(value: Ring48) -> usize {
        value.0 as usize
    }

    fn plaintext_less_than(d1: &(u16, u16), d2: &(u16, u16)) -> bool {
        let nmr1 = nhd_compare_nmr(d1.0, d1.1);
        let nmr2 = nhd_compare_nmr(d2.0, d2.1);
        nmr1 * (d2.1 as i64) < nmr2 * (d1.1 as i64)
    }

    fn plaintext_is_match(d: &(u16, u16)) -> bool {
        nhd_plaintext_is_match(d.0, d.1)
    }

    fn plaintext_ordering(d1: &(u16, u16), d2: &(u16, u16)) -> Ordering {
        let nmr1 = nhd_compare_nmr(d1.0, d1.1);
        let nmr2 = nhd_compare_nmr(d2.0, d2.1);
        (nmr1 * (d2.1 as i64)).cmp(&(nmr2 * (d1.1 as i64)))
    }

    fn plaintext_distance(a: &IrisCode, b: &IrisCode, f: DistanceFn) -> (u16, u16) {
        match f {
            DistanceFn::Simple => a.get_distance_fraction(b),
            DistanceFn::MinRotation => {
                a.get_min_nhd_distance_fraction_rotation_aware::<HAWK_MIN_ROTATIONS>(b)
            }
        }
    }
}
