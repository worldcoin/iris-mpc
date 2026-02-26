use crate::execution::session::Session;
use crate::protocol::ops::B;
use ampc_actor_utils::network::value::NetworkInt;
use ampc_actor_utils::protocol::{
    binary::{bit_inject, extract_msb_batch, open_bin},
    ops::{batch_signed_lift_vec_ring48, conditionally_select_distance},
};
use ampc_secret_sharing::shares::ring48::Ring48;
pub use ampc_secret_sharing::shares::{
    bit::Bit,
    ring_impl::RingElement,
    share::{DistanceShare, Share},
    vecshare::VecShare,
};
use eyre::Result;
use iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
use itertools::izip;
use tracing::instrument;

// ---------------------------------------------------------------------------
// NHD (Normalized Hamming Distance) protocol functions
// ---------------------------------------------------------------------------

/// NHD constants used in the polynomial approximation of the normalized distance.
/// The comparison `nhd(d1) < nhd(d2)` is reduced to checking the sign of
/// `nmr(d1)*md2 - nmr(d2)*md1` where
/// `nmr(d) = md*(md - 10*cd) - 73728*cd`.
const NHD_LINEAR_COEFF: u64 = 10;
const NHD_CORRECTION: u64 = 73728;

/// Computes the NHD cross product for comparing pairs of distances.
///
/// For each pair (d1, d2), computes `nmr(d1)*d2.md - nmr(d2)*d1.md` where
/// `nmr(d) = md*(md - 10*cd) - 73728*cd`.
///
/// This requires 2 interactive rounds:
/// - Round 1: compute `product_i = md_i * (md_i - 10*cd_i)` for each distance
/// - Round 2: compute the cross product `nmr_1*md_2 - nmr_2*md_1` and reshare
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn nhd_cross_mul(
    session: &mut Session,
    distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
) -> Result<Vec<Share<Ring48>>> {
    let n = distances.len();

    // ---- Round 1: interactive multiply to get product_i = md_i * linear_i ----
    // For each pair we compute 2 products (one per distance).
    let (prf_my_r1, prf_prev_r1) = session.prf.gen_rands_batch::<Ring48>(2 * n);

    let round1_a: Vec<RingElement<Ring48>> = distances
        .iter()
        .enumerate()
        .flat_map(|(i, (d1, d2))| {
            // linear_i = md_i - 10*cd_i
            let linear1 = d1.mask_dot - d1.code_dot * Ring48(NHD_LINEAR_COEFF);
            let linear2 = d2.mask_dot - d2.code_dot * Ring48(NHD_LINEAR_COEFF);

            // product_i = md_i * linear_i (local part of replicated multiplication)
            let prod1_a = prf_my_r1.0[2 * i] - prf_prev_r1.0[2 * i] + &d1.mask_dot * &linear1;
            let prod2_a =
                prf_my_r1.0[2 * i + 1] - prf_prev_r1.0[2 * i + 1] + &d2.mask_dot * &linear2;
            [prod1_a, prod2_a]
        })
        .collect();

    let network = &mut session.network_session;
    network
        .send_next(Ring48::new_network_vec(round1_a.clone()))
        .await?;
    let round1_b: Vec<RingElement<Ring48>> = Ring48::into_vec(network.receive_prev().await?)?;

    // Reconstruct product shares and compute nmr_i = product_i - 73728*cd_i
    let nmrs: Vec<(Share<Ring48>, Share<Ring48>)> = (0..n)
        .map(|i| {
            let prod1 = Share::new(round1_a[2 * i], round1_b[2 * i]);
            let prod2 = Share::new(round1_a[2 * i + 1], round1_b[2 * i + 1]);
            let nmr1 = prod1 - distances[i].0.code_dot * Ring48(NHD_CORRECTION);
            let nmr2 = prod2 - distances[i].1.code_dot * Ring48(NHD_CORRECTION);
            (nmr1, nmr2)
        })
        .collect();

    // ---- Round 2: cross product nmr_1*md_2 - nmr_2*md_1 and reshare ----
    let (prf_my_r2, prf_prev_r2) = session.prf.gen_rands_batch::<Ring48>(n);

    let round2_a: Vec<RingElement<Ring48>> = izip!(
        nmrs.iter(),
        distances.iter(),
        prf_my_r2.0.into_iter(),
        prf_prev_r2.0.into_iter()
    )
    .map(|((nmr1, nmr2), (d1, d2), my_r, prev_r)| {
        let zero_share = my_r - prev_r;
        zero_share + (nmr1 * &d2.mask_dot) - (nmr2 * &d1.mask_dot)
    })
    .collect();

    network
        .send_next(Ring48::new_network_vec(round2_a.clone()))
        .await?;
    let round2_b: Vec<RingElement<Ring48>> = Ring48::into_vec(network.receive_prev().await?)?;

    Ok(izip!(round2_a, round2_b)
        .map(|(a, b)| Share::new(a, b))
        .collect())
}

/// For every pair of NHD distance shares (d1, d2), computes d1 < d2 and opens it.
pub async fn nhd_cross_compare(
    session: &mut Session,
    distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
) -> Result<Vec<bool>> {
    let diff = nhd_cross_mul(session, distances).await?;
    let bits = extract_msb_batch(session, &diff).await?;
    let opened_b = open_bin(session, &bits).await?;
    opened_b.into_iter().map(|x| Ok(x.convert())).collect()
}

/// For every pair of NHD distance shares (d1, d2), computes the secret-shared bit d1 < d2.
pub async fn nhd_oblivious_cross_compare(
    session: &mut Session,
    distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
) -> Result<Vec<Share<Bit>>> {
    let diff = nhd_cross_mul(session, distances).await?;
    extract_msb_batch(session, &diff).await
}

/// For every pair of NHD distance shares (d1, d2), computes the secret-shared bit d1 < d2
/// and lifts it to Ring48 shares.
pub async fn nhd_oblivious_cross_compare_lifted(
    session: &mut Session,
    distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
) -> Result<Vec<Share<Ring48>>> {
    let bits = nhd_oblivious_cross_compare(session, distances).await?;
    Ok(bit_inject(session, VecShare { shares: bits })
        .await?
        .inner())
}

/// For every pair of NHD distance shares (d1, d2), returns the minimum distance.
pub async fn nhd_min_of_pair_batch(
    session: &mut Session,
    distances: &[(DistanceShare<Ring48>, DistanceShare<Ring48>)],
) -> Result<Vec<DistanceShare<Ring48>>> {
    let bits = nhd_oblivious_cross_compare_lifted(session, distances).await?;
    conditionally_select_distance(session, distances, bits.as_slice()).await
}

/// Lifts `u16` distance shares to `Ring48` and chunks them into `DistanceShare<Ring48>`.
pub async fn nhd_lift_distances(
    session: &mut Session,
    pre_lift: Vec<Share<u16>>,
) -> Result<Vec<DistanceShare<Ring48>>> {
    let lifted = batch_signed_lift_vec_ring48(session, pre_lift).await?;
    Ok(lifted
        .chunks(2)
        .map(|chunk| DistanceShare::new(chunk[0], chunk[1]))
        .collect())
}

/// Constant A used in the NHD threshold comparison.
/// Guaranteed to be positive if MATCH_THRESHOLD_RATIO <= 0.4725.
const A: u64 = 774_144_u64 - (25.0 * B as f64 * MATCH_THRESHOLD_RATIO) as u64;

/// Compares the distance between two iris pairs to a threshold.
///
/// - Takes as input a pair of code and mask dot products between two irises,
///   i.e., `cd = <iris1.code, iris2.code>` and `md = <iris1.mask, iris2.mask>`,
///   already lifted to 48 bits if they are originally 16-bit.
/// - Compares `md * (50 * cd - 5 * md) - A * md + 368_640 * cd < 0`.
/// - This corresponds to "distance > threshold", that is NOT match.
pub async fn nhd_greater_than_threshold(
    session: &mut Session,
    distances: &[DistanceShare<Ring48>],
) -> Result<Vec<Share<Bit>>> {
    let n = distances.len();

    // We check: [md * (50*cd - 5*md)] - A*md + 368640*cd < 0
    // The bracketed term requires one interactive multiplication round.

    let (prf_my, prf_prev) = session.prf.gen_rands_batch::<Ring48>(n);

    let round_a: Vec<RingElement<Ring48>> = distances
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let linear = d.code_dot * Ring48(50) - d.mask_dot * Ring48(5);
            prf_my.0[i] - prf_prev.0[i] + &d.mask_dot * &linear
        })
        .collect();

    let network = &mut session.network_session;
    network
        .send_next(Ring48::new_network_vec(round_a.clone()))
        .await?;
    let round_b: Vec<RingElement<Ring48>> = Ring48::into_vec(network.receive_prev().await?)?;

    let results: Vec<Share<Ring48>> = (0..n)
        .map(|i| {
            let product = Share::new(round_a[i], round_b[i]);
            product - distances[i].mask_dot * Ring48(A) + distances[i].code_dot * Ring48(368640)
        })
        .collect();

    extract_msb_batch(session, &results).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{execution::local::LocalRuntime, shares::int_ring::IntRing2k};
    use aes_prng::AesRng;
    use ampc_actor_utils::protocol::ops::batch_signed_lift_vec_ring48;
    use rand::{Rng, RngCore, SeedableRng};
    use rand_distr::{Distribution, Standard};
    use tokio::task::JoinSet;

    fn create_single_sharing<R: RngCore, T: IntRing2k>(
        rng: &mut R,
        input: T,
    ) -> (Share<T>, Share<T>, Share<T>)
    where
        Standard: Distribution<T>,
    {
        let a = RingElement(rng.gen::<T>());
        let b = RingElement(rng.gen::<T>());
        let c = RingElement(input) - a - b;

        let share1 = Share::new(a, c);
        let share2 = Share::new(b, a);
        let share3 = Share::new(c, b);
        (share1, share2, share3)
    }

    type ThreePartyShares<T> = (Vec<Share<T>>, Vec<Share<T>>, Vec<Share<T>>);

    fn create_array_sharing<R: RngCore, T: IntRing2k>(
        rng: &mut R,
        input: &[T],
    ) -> ThreePartyShares<T>
    where
        Standard: Distribution<T>,
    {
        let mut player0 = Vec::new();
        let mut player1 = Vec::new();
        let mut player2 = Vec::new();

        for entry in input {
            let (a, b, c) = create_single_sharing(rng, *entry);
            player0.push(a);
            player1.push(b);
            player2.push(c);
        }
        (player0, player1, player2)
    }

    fn reference_nhd(cd: i64, md: i64) -> f64 {
        if md == 0 {
            return f64::INFINITY; // Define NHD as infinity when mask dot product is zero to avoid division by zero
        }
        let cd_f = cd as f64;
        let md_f = md as f64;
        0.45 - (0.45 - (md_f - cd_f) / (2.0 * md_f)) * (md_f / 16384.0 + 0.45)
    }

    /// Reference plaintext NHD comparison: returns true if nhd(d1) < nhd(d2), i.e., d1 is
    /// a better match.
    fn reference_nhd_less_than(cd1: i64, md1: i64, cd2: i64, md2: i64) -> bool {
        let dist1 = reference_nhd(cd1, md1);
        let dist2 = reference_nhd(cd2, md2);
        dist1 < dist2
    }

    /// Reference plaintext NHD threshold check: returns true if the NHD distance is greater
    /// than the threshold (NOT match).
    fn reference_nhd_greater_than_threshold(cd: i64, md: i64) -> bool {
        reference_nhd(cd, md) > MATCH_THRESHOLD_RATIO
    }

    /// Convert negative values of `cd` represented in u16 to their signed i64 representation
    fn convert_to_signed(cd: u16) -> i64 {
        if cd > (1 << 15) {
            cd as i64 - (1 << 16)
        } else {
            cd as i64
        }
    }

    #[tokio::test]
    async fn test_nhd_greater_than_threshold() {
        let mut rng = AesRng::seed_from_u64(43_u64);

        // Test with known values of `(code_dot, mask_dot)` and their expected NHD threshold comparison result.
        // It should hold that `abs(code dot) < mask_dot`.
        // Expected boolean is computed as `nhd > MATCH_THRESHOLD_RATIO`.
        let test_cases: Vec<(u16, u16, bool)> = vec![
            (100, 500, true),  // nhd = 0.43
            (400, 500, false), // nhd = 0.28
            // Edge cases around the threshold with the same FHD
            (1300, 3000, false), // nhd = 0.34
            (13, 30, true),      // nhd = 0.37
            // Edge cases
            (0, 100, true),             // nhd = 0.47
            (u16::MAX, 200, true),      // (-1, 200) -> nhd = 0.47
            (u16::MAX, 1, true),        // (-1, 1) -> nhd = 0.70
            (u16::MAX - 99, 100, true), // (-100, 100) -> nhd = 0.70
            (1, 1, false),              // nhd = 0.25
        ];

        let flat_values: Vec<u16> = test_cases
            .iter()
            .flat_map(|(cd, md, _)| [*cd, *md])
            .collect();
        let (p0, p1, p2) = create_array_sharing(&mut rng, &flat_values);

        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut jobs = JoinSet::new();

        for (i, session) in sessions.into_iter().enumerate() {
            let session = session.clone();
            let shares_i = match i {
                0 => p0.clone(),
                1 => p1.clone(),
                2 => p2.clone(),
                _ => unreachable!(),
            };
            let n = test_cases.len();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                let lifted = batch_signed_lift_vec_ring48(&mut session, shares_i)
                    .await
                    .unwrap();
                let distances: Vec<DistanceShare<Ring48>> = (0..n)
                    .map(|j| DistanceShare::new(lifted[2 * j], lifted[2 * j + 1]))
                    .collect();
                let bits = nhd_greater_than_threshold(&mut session, &distances)
                    .await
                    .unwrap();
                let opened = open_bin(&mut session, &bits).await.unwrap();
                opened
                    .into_iter()
                    .map(|x| x.convert())
                    .collect::<Vec<bool>>()
            });
        }

        let results: Vec<Vec<bool>> = jobs.join_all().await;

        // All parties should agree
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);

        // Check against plaintext
        for (i, (cd, md, expected)) in test_cases.into_iter().enumerate() {
            let ref_cd = convert_to_signed(cd);
            let reference = reference_nhd_greater_than_threshold(ref_cd, md as i64);
            assert_eq!(
                results[0][i], reference,
                "Reference NHD threshold mismatch for (cd={}, md={}): got {}, expected {}",
                cd, md, results[0][i], reference
            );
            assert_eq!(
                results[0][i], expected,
                "NHD threshold mismatch for (cd={}, md={}): got {}, expected {}",
                cd, md, results[0][i], expected
            );
        }
    }

    #[tokio::test]
    async fn test_nhd_cross_compare() {
        let mut rng = AesRng::seed_from_u64(42_u64);

        // These represent (code_dot1, mask_dot1, code_dot2, mask_dot2, expected result) pairs.
        let test_cases: Vec<(u16, u16, u16, u16, bool)> = vec![
            (100, 500, 200, 600, false),                       // 0.43 > 0.39
            (400, 500, 200, 600, true),                        // 0.28 < 0.39
            (u16::MAX - 399, 500, u16::MAX - 199, 600, false), // (-400, 500) vs (-200, 600) -> 0.67 > 0.56
            // same FHD but different mask dot
            (1300, 3000, 13, 30, true),                // 0.34 < 0.37
            (u16::MAX, 1, u16::MAX - 999, 1000, true), // 0.70 < 0.73
            // Edge cases
            (0, 0, 0, 0, false), // nhd = infinity vs infinity -> no greater than
            (0, 100, 0, 100, false), // nhd = 0.47 vs 0.47 -> no greater than
        ];

        // Create shares of all values: [cd1, md1, cd2, md2, ...]
        let flat_values: Vec<u16> = test_cases
            .iter()
            .flat_map(|(cd1, md1, cd2, md2, _)| [*cd1, *md1, *cd2, *md2])
            .collect();
        let (p0, p1, p2) = create_array_sharing(&mut rng, &flat_values);

        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut jobs = JoinSet::new();

        for (i, session) in sessions.into_iter().enumerate() {
            let session = session.clone();
            let shares_i = match i {
                0 => p0.clone(),
                1 => p1.clone(),
                2 => p2.clone(),
                _ => unreachable!(),
            };
            let n = test_cases.len();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                // Lift u16 shares to Ring48
                let lifted = batch_signed_lift_vec_ring48(&mut session, shares_i)
                    .await
                    .unwrap();
                // Build distance pairs
                let pairs: Vec<(DistanceShare<Ring48>, DistanceShare<Ring48>)> = (0..n)
                    .map(|j| {
                        let d1 = DistanceShare::new(lifted[4 * j], lifted[4 * j + 1]);
                        let d2 = DistanceShare::new(lifted[4 * j + 2], lifted[4 * j + 3]);
                        (d1, d2)
                    })
                    .collect();
                nhd_cross_compare(&mut session, &pairs).await.unwrap()
            });
        }

        let results: Vec<Vec<bool>> = jobs.join_all().await;

        // All parties should agree
        assert_eq!(results[0], results[1]);
        assert_eq!(results[1], results[2]);

        // Check against plaintext
        for (i, (cd1, md1, cd2, md2, expected)) in test_cases.into_iter().enumerate() {
            let ref_cd1 = convert_to_signed(cd1);
            let ref_cd2 = convert_to_signed(cd2);
            let reference = reference_nhd_less_than(ref_cd1, md1 as i64, ref_cd2, md2 as i64);
            assert_eq!(
                results[0][i], reference,
                "Reference NHD comparison mismatch for ({}, {}) vs ({}, {}): got {}, expected {}",
                cd1, md1, cd2, md2, results[0][i], reference
            );
            assert_eq!(
                results[0][i], expected,
                "NHD comparison mismatch for ({}, {}) vs ({}, {}): got {}, expected {}",
                cd1, md1, cd2, md2, results[0][i], expected
            );
        }
    }
}
