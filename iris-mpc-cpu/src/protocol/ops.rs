use super::binary::{
    bit_inject_ot_2round, extract_msb_u32_batch, lift, mul_lift_2k, open_bin,
    single_extract_msb_u32,
};
use crate::{
    execution::session::{NetworkSession, Session, SessionHandles},
    network::value::{
        NetworkInt,
        NetworkValue::{self},
    },
    protocol::{
        prf::{Prf, PrfSeed},
        shared_iris::ArcIris,
    },
    shares::{
        bit::Bit,
        ring_impl::{RingElement, VecRingElement},
        share::{DistanceShare, Share},
        vecshare::VecShare,
        IntRing2k,
    },
};
use eyre::{bail, eyre, Result};
use iris_mpc_common::{
    fast_metrics::FastHistogram,
    galois_engine::degree4::{IrisRotation, SHARE_OF_MAX_DISTANCE},
    ROTATIONS,
};
use itertools::{izip, Itertools};
use std::{array, ops::Not, time::Instant};
use tracing::instrument;

pub(crate) const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub(crate) const B_BITS: u64 = 16;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;

/// Setup the PRF seeds in the replicated protocol.
/// Each party sends to the next party a random seed.
/// At the end, each party will hold two seeds which are the basis of the
/// replicated protocols.
#[instrument(
    level = "trace",
    target = "searcher::network",
    fields(party = ?session.own_role),
    skip_all
)]
pub async fn setup_replicated_prf(session: &mut NetworkSession, my_seed: PrfSeed) -> Result<Prf> {
    // send my_seed to the next party
    session.send_next(NetworkValue::PrfKey(my_seed)).await?;
    // deserializing received seed.
    let other_seed = match session.receive_prev().await {
        Ok(NetworkValue::PrfKey(seed)) => seed,
        _ => bail!("Could not deserialize PrfKey"),
    };
    // creating the two PRFs
    Ok(Prf::new(my_seed, other_seed))
}

/// Setup an RNG common between all parties, for use in stochastic algorithms (e.g. HNSW layer selection).
pub async fn setup_shared_seed(session: &mut NetworkSession, my_seed: PrfSeed) -> Result<PrfSeed> {
    let my_msg = NetworkValue::PrfKey(my_seed);

    let decode = |msg| match msg {
        Ok(NetworkValue::PrfKey(seed)) => Ok(seed),
        _ => Err(eyre!("Could not deserialize PrfKey")),
    };

    // Round 1: Send to the next party and receive from the previous party.
    session.send_next(my_msg.clone()).await?;
    let prev_seed = decode(session.receive_prev().await)?;

    // Round 2: Send/receive in the opposite direction.
    session.send_prev(my_msg).await?;
    let next_seed = decode(session.receive_next().await)?;

    let shared_seed = array::from_fn(|i| my_seed[i] ^ prev_seed[i] ^ next_seed[i]);
    Ok(shared_seed)
}

/// Compares the distance between two iris pairs to a threshold.
///
/// - Takes as input two code and mask dot products between two irises,
///   i.e., code_dist = <iris1.code, iris2.code> and mask_dist = <iris1.mask, iris2.mask>.
/// - Lifts the two dot products to the ring Z_{2^32}.
/// - Multiplies with predefined threshold constants B = 2^16 and A = ((1. - 2.
///   * MATCH_THRESHOLD_RATIO) * B as f64).
/// - Compares mask_dist * A > code_dist * B.
/// - This corresponds to "distance > threshold", that is NOT match.
pub async fn greater_than_threshold(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
) -> Result<Vec<Share<Bit>>> {
    let diffs: Vec<Share<u32>> = distances
        .iter()
        .map(|d| {
            let x = d.mask_dot.clone() * A as u32;
            let y = d.code_dot.clone() * B as u32;
            y - x
        })
        .collect();

    extract_msb_u32_batch(session, &diffs).await
}

/// Computes the `A` term of the threshold comparison based on the formula `A = ((1. - 2. * t) * B)`.
pub fn translate_threshold_a(t: f64) -> u32 {
    assert!(
        (0. ..=1.).contains(&t),
        "Threshold must be in the range [0, 1]"
    );
    ((1. - 2. * t) * (B as f64)) as u32
}

/// Compares the distance between two iris pairs to a list of thresholds, represented as t_i/B, with B = 2^16.
/// Use the [translate_threshold_a] function to compute the A term of the threshold comparison.
/// The result of the comparisons is then summed up bucket-wise, with each bucket corresponding to a threshold.
pub async fn compare_threshold_buckets(
    session: &mut Session,
    threshold_a_terms: &[u32],
    distances: &[DistanceShare<u32>],
) -> Result<Vec<Share<u32>>> {
    let diffs = threshold_a_terms
        .iter()
        .flat_map(|a| {
            distances.iter().map(|d| {
                let x = d.mask_dot.clone() * *a;
                let y = d.code_dot.clone() * B as u32;
                x - y
            })
        })
        .collect_vec();

    tracing::info!("compare_threshold_buckets diffs length: {}", diffs.len());
    let msbs = extract_msb_u32_batch(session, &diffs).await?;
    let msbs = VecShare::new_vec(msbs);
    tracing::info!("msbs extracted, now bit_injecting");
    // bit_inject all MSBs into u32 to be able to add them up
    let sums = bit_inject_ot_2round(session, msbs).await?;
    tracing::info!("bit_inject done, now summing");
    // add them up, bucket-wise, with each bucket corresponding to a threshold and containing len(distances) results
    let buckets = sums
        .into_iter()
        .chunks(distances.len())
        .into_iter()
        .map(|chunk| chunk.reduce(|a, b| a + b).unwrap_or_default())
        .collect_vec();

    Ok(buckets)
}

/// Compares the distance between two iris pairs to a list of thresholds, represented as t_i/B, with B = 2^16.
/// Use the [translate_threshold_a] function to compute the A term of the threshold comparison.
/// The result of the comparisons is then summed up bucket-wise, with each bucket corresponding to a threshold.
///
/// In comparison to `compare_threshold_buckets`, this function takes grouped distances as input, and for each group
/// only the minimum distance is considered for creating the buckets.
pub async fn compare_min_threshold_buckets(
    session: &mut Session,
    threshold_a_terms: &[u32],
    distances: &[Vec<DistanceShare<u32>>],
) -> Result<Vec<Share<u32>>> {
    // grab the first one of the distance in each group
    let mut reduced_distances = distances
        .iter()
        .map(|group| {
            group
                .first()
                .cloned()
                .ok_or_else(|| eyre!("Expected at least one distance in the group"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sizes = distances
        .iter()
        .map(|group| group.len() - 1)
        .collect::<Vec<_>>();

    // This loop is executed at most MAX_ROTATIONS-1 times, which is 30 currently
    // however, in practice it will probably be executed much less often.
    while !sizes.iter().all(|&size| size == 0) {
        // we grab a vector of potential rotations to reduce
        // If this current group is already reduced to 0, we grab the first element as a dummy copy.
        let distances_to_reduce: Vec<(DistanceShare<u32>, DistanceShare<u32>)> = distances
            .iter()
            .zip(sizes.iter_mut())
            .map(|(group, size)| {
                let element_to_reduce = group[*size].clone();
                if *size > 0 {
                    *size -= 1;
                }
                element_to_reduce
            })
            .zip(reduced_distances.iter())
            .map(|(new, reduced)| (new, reduced.clone()))
            .collect();

        reduced_distances = min_of_pair_batch(session, &distances_to_reduce).await?;
    }

    // Now we have a single distance for each group, we can compare it to the thresholds
    let buckets = compare_threshold_buckets(session, threshold_a_terms, &reduced_distances).await?;

    Ok(buckets)
}

/// The same as compare_threshold, but the input shares are 16-bit and lifted to
/// 32-bit before threshold comparison.
///
/// See compare_threshold for more details.
pub async fn lift_and_compare_threshold(
    session: &mut Session,
    code_dist: Share<u16>,
    mask_dist: Share<u16>,
) -> Result<Share<Bit>> {
    let mut y = mul_lift_2k::<B_BITS>(&code_dist);
    let mut x = lift(session, VecShare::new_vec(vec![mask_dist])).await?;
    let mut x = x
        .pop()
        .ok_or(eyre!("Expected a single element in the VecShare"))?;
    x *= A as u32;
    y -= x;

    single_extract_msb_u32(session, y).await
}

/// Lifts a share of a vector (VecShare) of 16-bit values to a share of a vector
/// (VecShare) of 32-bit values.
pub async fn batch_signed_lift(
    session: &mut Session,
    mut pre_lift: VecShare<u16>,
) -> Result<VecShare<u32>> {
    // Compute (v + 2^{15}) % 2^{16}, to make values positive.
    for v in pre_lift.iter_mut() {
        v.add_assign_const_role(1_u16 << 15, session.own_role());
    }
    let mut lifted_values = lift(session, pre_lift).await?;
    // Now we got shares of d1' over 2^32 such that d1' = (d1'_1 + d1'_2 + d1'_3) %
    // 2^{16} = d1 Next we subtract the 2^15 term we've added previously to
    // get signed shares over 2^{32}
    for v in lifted_values.iter_mut() {
        v.add_assign_const_role(((1_u64 << 32) - (1_u64 << 15)) as u32, session.own_role());
    }
    Ok(lifted_values)
}

/// Wrapper over batch_signed_lift that lifts a vector (Vec) of 16-bit shares to
/// a vector (Vec) of 32-bit shares.
pub async fn batch_signed_lift_vec(
    session: &mut Session,
    pre_lift: Vec<Share<u16>>,
) -> Result<Vec<Share<u32>>> {
    let pre_lift = VecShare::new_vec(pre_lift);
    Ok(batch_signed_lift(session, pre_lift).await?.inner())
}

/// Computes the cross product of distances shares represented as a fraction (code_dist, mask_dist).
/// The cross product is computed as (d2.code_dist * d1.mask_dist - d1.code_dist * d2.mask_dist) and the result is shared.
///
/// Assumes that the input shares are originally 16-bit and lifted to u32.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn cross_mul(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
) -> Result<Vec<Share<u32>>> {
    let res_a: VecRingElement<u32> = distances
        .iter()
        .map(|(d1, d2)| {
            session.prf.gen_zero_share() + &d2.code_dot * &d1.mask_dot - &d1.code_dot * &d2.mask_dot
        })
        .collect();

    let network = &mut session.network_session;

    network.send_ring_vec_next(&res_a).await?;

    let res_b = network.receive_ring_vec_prev().await?;
    Ok(izip!(res_a, res_b).map(|(a, b)| Share::new(a, b)).collect())
}

/// Conditionally selects equally-sized slices of input shares based on control bits.
/// If the control bit is 1, it selects the left value shares; otherwise, it selects the right value share.
async fn select_shared_slices_by_bits(
    session: &mut Session,
    left_values: &[Share<u32>],
    right_values: &[Share<u32>],
    control_bits: &[Share<u32>],
    slice_size: usize,
) -> Result<Vec<Share<u32>>> {
    if left_values.len() != right_values.len() {
        bail!("Left and right values must have the same length");
    }
    if left_values.len() % slice_size != 0 {
        bail!("Left and right values length must be multiple of slice size");
    }
    if control_bits.len() != left_values.len() / slice_size {
        bail!("Number of control bits must match number of slices");
    }

    // Conditional multiplexing:
    // If control bit is 1, select left_value, else select right_value.
    // res = c * (left_value - right_value) + right_value
    // Compute c * (left_value - right_value)
    let res_a: VecRingElement<u32> = izip!(
        left_values.chunks(slice_size),
        right_values.chunks(slice_size),
        control_bits.iter()
    )
    .flat_map(|(left_chunk, right_chunk, c)| {
        left_chunk
            .iter()
            .zip(right_chunk.iter())
            .map(|(left, right)| {
                let diff = left.clone() - right.clone();
                session.prf.gen_zero_share() + c.a * diff.a + c.b * diff.a + c.a * diff.b
            })
            .collect_vec()
    })
    .collect();

    let network = &mut session.network_session;

    network.send_ring_vec_next(&res_a).await?;

    let res_b = network.receive_ring_vec_prev().await?;

    // Pack networking messages into shares and
    // compute the result by adding the right shares
    Ok(izip!(res_a, res_b)
        .map(|(a, b)| Share::new(a, b))
        .zip(right_values.iter())
        .map(|(res, right)| res + right)
        .collect())
}

/// Conditionally selects the distance shares based on control bits.
/// If the control bit is 1, it selects the first distance share (d1),
/// otherwise it selects the second distance share (d2).
/// Assumes that the input shares are originally 16-bit and lifted to u32.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn conditionally_select_distance(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
    control_bits: &[Share<u32>],
) -> Result<Vec<DistanceShare<u32>>> {
    if distances.len() != control_bits.len() {
        bail!("Number of distances must match number of control bits");
    }

    // Conditional multiplexing:
    // If control bit is 1, select d1, else select d2.
    // res = c * d1 + (1 - c) * d2 = d2 + c * (d1 - d2);
    // We need to do it for both code_dot and mask_dot.

    // we start with the mult of c and d1-d2
    let res_a: VecRingElement<u32> = distances
        .iter()
        .zip(control_bits.iter())
        .flat_map(|((d1, d2), c)| {
            let code = d1.code_dot.clone() - d2.code_dot.clone();
            let mask = d1.mask_dot.clone() - d2.mask_dot.clone();
            let code_mul_a =
                session.prf.gen_zero_share() + c.a * code.a + c.b * code.a + c.a * code.b;
            let mask_mul_a =
                session.prf.gen_zero_share() + c.a * mask.a + c.b * mask.a + c.a * mask.b;
            [code_mul_a, mask_mul_a]
        })
        .collect();

    let network = &mut session.network_session;

    network.send_ring_vec_next(&res_a).await?;

    let res_b = network.receive_ring_vec_prev().await?;

    // finally compute the result by adding the d2 shares
    Ok(izip!(res_a, res_b)
        // combine a and b part into shares
        .map(|(a, b)| Share::new(a, b))
        // combine the code and mask parts into DistanceShare
        .tuples()
        .map(|(code, mask)| DistanceShare {
            code_dot: code,
            mask_dot: mask,
        })
        // add the d2 shares
        .zip(distances.iter())
        .map(|(res, (_, d2))| DistanceShare {
            code_dot: res.code_dot + &d2.code_dot,
            mask_dot: res.mask_dot + &d2.mask_dot,
        })
        .collect())
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn conditionally_select_distances_with_plain_ids(
    session: &mut Session,
    left_distances: Vec<(u32, DistanceShare<u32>)>,
    right_distances: Vec<(u32, DistanceShare<u32>)>,
    control_bits: Vec<Share<u32>>,
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    if left_distances.len() != control_bits.len() {
        eyre::bail!("Number of distances must match number of control bits");
    }
    if left_distances.len() != right_distances.len() {
        eyre::bail!("Left and right distances must have the same length");
    }
    if left_distances.is_empty() {
        eyre::bail!("Distances must not be empty");
    }

    // Now select distances
    let (left_ids, left_dist): (Vec<_>, Vec<_>) = left_distances.into_iter().unzip();
    let (right_ids, right_dist): (Vec<_>, Vec<_>) = right_distances.into_iter().unzip();
    let left_dist = left_dist
        .into_iter()
        .flat_map(|d| [d.code_dot, d.mask_dot])
        .collect_vec();
    let right_dist = right_dist
        .into_iter()
        .flat_map(|d| [d.code_dot, d.mask_dot])
        .collect_vec();

    let distances =
        select_shared_slices_by_bits(session, &left_dist, &right_dist, &control_bits, 2)
            .await?
            .into_iter()
            .tuples()
            .map(|(code_dot, mask_dot)| DistanceShare::new(code_dot, mask_dot));

    // Select ids first: c * (left_id - right_id) + right_id
    let ids = izip!(left_ids, right_ids, control_bits).map(|(left_id, right_id, c)| {
        let diff = left_id.wrapping_sub(right_id);
        let mut res = c.clone() * RingElement(diff);
        res.add_assign_const_role(right_id, session.own_role());
        res
    });

    Ok(izip!(ids, distances)
        .map(|(id, distance)| (id, distance))
        .collect_vec())
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub(crate) async fn conditionally_select_distances_with_shared_ids(
    session: &mut Session,
    left_distances: Vec<(Share<u32>, DistanceShare<u32>)>,
    right_distances: Vec<(Share<u32>, DistanceShare<u32>)>,
    control_bits: Vec<Share<u32>>,
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    if left_distances.len() != control_bits.len() {
        eyre::bail!("Number of distances must match number of control bits");
    }
    if left_distances.len() != right_distances.len() {
        eyre::bail!("Left and right distances must have the same length");
    }
    if left_distances.is_empty() {
        eyre::bail!("Distances must not be empty");
    }

    let left_dist = left_distances
        .into_iter()
        .flat_map(|(id, d)| [id, d.code_dot.clone(), d.mask_dot.clone()])
        .collect_vec();
    let right_dist = right_distances
        .into_iter()
        .flat_map(|(id, d)| [id, d.code_dot.clone(), d.mask_dot.clone()])
        .collect_vec();
    let distances =
        select_shared_slices_by_bits(session, &left_dist, &right_dist, &control_bits, 3)
            .await?
            .into_iter()
            .tuples()
            .map(|(id, code_dot, mask_dot)| (id, DistanceShare::new(code_dot, mask_dot)))
            .collect_vec();

    Ok(distances)
}

/// Conditionally swaps the distance shares based on control bits.
/// Given the ith pair of indices (i1, i2), the function does the following.
/// If the control bit is 0, it swaps tuples (32-bit id, distance share) with index i1 and i2,
/// otherwise it does nothing.
/// Assumes that the input shares are originally 16-bit and lifted to u32.
/// The vector ids are in plaintext and propagated in secret shared form.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn conditionally_swap_distances_plain_ids(
    session: &mut Session,
    swap_bits: Vec<Share<Bit>>,
    list: &[(u32, DistanceShare<u32>)],
    indices: &[(usize, usize)],
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    if swap_bits.len() != indices.len() {
        eyre::bail!("swap bits and indices must have the same length");
    }
    let role = session.own_role();
    // Convert vector ids into trivial shares
    let mut encrypted_list = list
        .iter()
        .map(|(id, d)| {
            let shared_index = Share::from_const(*id, role);
            (shared_index, d.clone())
        })
        .collect_vec();
    // Lift swap bits to u32 shares
    let swap_bits_u32 = bit_inject_ot_2round(session, VecShare::<Bit>::new_vec(swap_bits))
        .await?
        .inner();

    let distances_to_swap = indices
        .iter()
        .filter_map(|(idx1, idx2)| match (list.get(*idx1), list.get(*idx2)) {
            (Some((_, d1)), Some((_, d2))) => Some((d1.clone(), d2.clone())),
            _ => None,
        })
        .collect_vec();
    // Select the first distance in each pair based on the control bits
    let first_distances =
        conditionally_select_distance(session, &distances_to_swap, &swap_bits_u32).await?;
    // Select the second distance in each pair as sum of both distances minus the first selected distance
    let second_distances = distances_to_swap
        .into_iter()
        .zip(first_distances.iter())
        .map(|(d_pair, first_d)| {
            DistanceShare::new(
                d_pair.0.code_dot + d_pair.1.code_dot - &first_d.code_dot,
                d_pair.0.mask_dot + d_pair.1.mask_dot - &first_d.mask_dot,
            )
        })
        .collect_vec();

    for (bit, (idx1, idx2), first_d, second_d) in izip!(
        swap_bits_u32.iter(),
        indices.iter(),
        first_distances,
        second_distances
    ) {
        let mut not_bit = -bit;
        not_bit.add_assign_const_role(1, role);
        let id1 = list[*idx1].0;
        let id2 = list[*idx2].0;
        // Only propagate index and skip version id.
        // This computation is local as indices are public.
        let first_id = bit * id1 + not_bit.clone() * id2;
        let second_id = bit * id2 + not_bit * id1;
        encrypted_list[*idx1] = (first_id, first_d);
        encrypted_list[*idx2] = (second_id, second_d);
    }
    Ok(encrypted_list)
}

/// Conditionally swaps the distance shares based on control bits.
/// Given the ith pair of indices (i1, i2), the function does the following.
/// If the ith control bit is 0, it swaps tuples (0-indexed vector id, distance share) with index i1 and i2,
/// otherwise it does nothing.
/// Assumes that the input shares are originally 16-bit and lifted to u32.
/// The vector ids are 0-indexed and given in secret shared form.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn conditionally_swap_distances(
    session: &mut Session,
    swap_bits: Vec<Share<Bit>>,
    list: &[(Share<u32>, DistanceShare<u32>)],
    indices: &[(usize, usize)],
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    if swap_bits.len() != indices.len() {
        return Err(eyre!("swap bits and indices must have the same length"));
    }
    // Lift bits to u32 shares
    let swap_bits_u32 = bit_inject_ot_2round(session, VecShare::<Bit>::new_vec(swap_bits))
        .await?
        .inner();

    // A helper closure to compute the difference of two input shares and prepare the a part of the product of this difference and the control bit.
    let mut mul_share_a = |x: Share<u32>, y: Share<u32>, sb: &Share<u32>| -> RingElement<u32> {
        let diff = x - y;
        session.prf.gen_zero_share() + sb.a * diff.a + sb.b * diff.a + sb.a * diff.b
    };

    // Conditional swapping:
    // If control bit c is 1, return (d1, d2); otherwise, (d2, d1), which can be computed as:
    // - first tuple element = c * (d1 - d2) + d2;
    // - second tuple element = d1 - c * (d1 - d2).
    // We need to do it for ids, code_dot and mask_dot.

    // Compute c * (d1-d2)
    let res_a: VecRingElement<u32> = indices
        .iter()
        .zip(swap_bits_u32.iter())
        .flat_map(|((idx1, idx2), sb)| {
            let (id1, d1) = &list[*idx1];
            let (id2, d2) = &list[*idx2];

            let id = mul_share_a(id1.clone(), id2.clone(), sb);
            let code_dot_a = mul_share_a(d1.code_dot.clone(), d2.code_dot.clone(), sb);
            let mask_dot_a = mul_share_a(d1.mask_dot.clone(), d2.mask_dot.clone(), sb);
            [id, code_dot_a, mask_dot_a]
        })
        .collect();

    let network = &mut session.network_session;

    network.send_ring_vec_next(&res_a).await?;

    let res_b = network.receive_ring_vec_prev().await?;

    // Finally compute the swapped tuples.
    let swapped_distances = izip!(res_a, res_b)
        // combine a and b part into shares
        .map(|(a, b)| Share::new(a, b))
        // combine the code and mask parts into DistanceShare
        .tuples()
        .map(|(id, code, mask)| {
            (
                id,
                DistanceShare {
                    code_dot: code,
                    mask_dot: mask,
                },
            )
        })
        .zip(indices.iter())
        .map(|((res_id, res_dist), (idx1, idx2))| {
            let (id1, dist1) = &list[*idx1];
            let (id2, dist2) = &list[*idx2];
            // first tuple element = c * (d1 - d2) + d2
            // second tuple element = d1 - c * (d1 - d2)
            let first_id = res_id.clone() + id2;
            let second_id = id1.clone() - res_id;
            let first_distance = DistanceShare {
                code_dot: res_dist.code_dot.clone() + &dist2.code_dot,
                mask_dot: res_dist.mask_dot.clone() + &dist2.mask_dot,
            };
            let second_distance = DistanceShare {
                code_dot: dist1.code_dot.clone() - res_dist.code_dot,
                mask_dot: dist1.mask_dot.clone() - res_dist.mask_dot,
            };
            ((first_id, first_distance), (second_id, second_distance))
        })
        .collect_vec();

    // Update the input list with the swapped tuples.
    let mut swapped_list = list.to_vec();
    for (((id1, d1), (id2, d2)), (idx1, idx2)) in swapped_distances.into_iter().zip(indices) {
        swapped_list[*idx1] = (id1, d1);
        swapped_list[*idx2] = (id2, d2);
    }

    Ok(swapped_list)
}

/// For every pair of distance shares (d1, d2), this computes the bit d2 < d1 and opens it.
///
/// The less-than operator is implemented in 2 steps:
///
/// 1. d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot is computed, which is a numerator of the fraction difference d2.code_dot / d2.mask_dot - d1.code_dot / d1.mask_dot.
/// 2. The most significant bit of the result is extracted.
///
/// Input values are assumed to be 16-bit shares that have been lifted to 32 bits.
pub async fn cross_compare(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
) -> Result<Vec<bool>> {
    // d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot
    let diff = cross_mul(session, distances).await?;
    // Compute the MSB of the above
    let bits = extract_msb_u32_batch(session, &diff).await?;
    // Open the MSB
    let opened_b = open_bin(session, &bits).await?;
    opened_b.into_iter().map(|x| Ok(x.convert())).collect()
}

/// For every pair of distance shares (d1, d2), this computes the secret-shared bit d2 < d1 .
///
/// The less-than operator is implemented in 2 steps:
///
/// 1. d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot is computed, which is a numerator of the fraction difference d2.code_dot / d2.mask_dot - d1.code_dot / d1.mask_dot.
/// 2. The most significant bit of the result is extracted.
///
/// Input values are assumed to be 16-bit shares that have been lifted to 32 bits.
pub(crate) async fn oblivious_cross_compare(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
) -> Result<Vec<Share<Bit>>> {
    // d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot
    let diff = cross_mul(session, distances).await?;
    // Compute the MSB of the above
    extract_msb_u32_batch(session, &diff).await
}

/// For every pair of distance shares (d1, d2), this computes the secret-shared bit d2 < d1 and lift it to u32 shares.
///
/// The less-than operator is implemented in 2 steps:
///
/// 1. d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot is computed, which is a numerator of the fraction difference d2.code_dot / d2.mask_dot - d1.code_dot / d1.mask_dot.
/// 2. The most significant bit of the result is extracted.
///
/// Input values are assumed to be 16-bit shares that have been lifted to 32 bits.
pub(crate) async fn oblivious_cross_compare_lifted(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
) -> Result<Vec<Share<u32>>> {
    // compute the secret-shared bits d1 < d2
    let bits = oblivious_cross_compare(session, distances).await?;
    // inject bits to T shares
    Ok(bit_inject_ot_2round(session, VecShare { shares: bits })
        .await?
        .inner())
}

/// For every pair of distance shares (d1, d2), this computes the bit d2 < d1 uses it to return the lower of the two distances.
///
/// Input values are assumed to be 16-bit shares that have been lifted to 32 bits.
pub async fn min_of_pair_batch(
    session: &mut Session,
    distances: &[(DistanceShare<u32>, DistanceShare<u32>)],
) -> Result<Vec<DistanceShare<u32>>> {
    // compute the secret-shared bits d1 < d2
    let bits = oblivious_cross_compare_lifted(session, distances).await?;

    conditionally_select_distance(session, distances, bits.as_slice()).await
}

use std::cell::RefCell;

thread_local! {
    static PAIRWISE_DISTANCE_METRICS: RefCell<[FastHistogram; 2]> = RefCell::new([
        FastHistogram::new("pairwise_distance.batch_size"),
        FastHistogram::new("pairwise_distance.per_pair_duration"),
    ]);
}

/// See pairwise_distance.
/// This variant takes as input a Vec of Arc.
pub fn galois_ring_pairwise_distance(
    pairs: Vec<Option<(ArcIris, ArcIris)>>,
) -> Vec<RingElement<u16>> {
    pairwise_distance(pairs.iter().map(|opt| opt.as_ref().map(|(x, y)| (x, y))))
}

/// Computes the dot product between the iris pairs; for both the code and the
/// mask of the irises. We pack the dot products of the code and mask into one
/// vector to be able to reshare it later.
/// This function takes an iterator of known size.
pub fn pairwise_distance<'a, I>(pairs: I) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<(&'a ArcIris, &'a ArcIris)>> + ExactSizeIterator,
{
    let start = Instant::now();
    let mut count = 0;
    let mut additive_shares = Vec::with_capacity(2 * pairs.len());

    for pair in pairs {
        let (code_dist, mask_dist) = if let Some((x, y)) = pair {
            count += 1;
            let (a, b) = (x.code.trick_dot(&y.code), x.mask.trick_dot(&y.mask));
            (RingElement(a), RingElement(2) * RingElement(b))
        } else {
            // Non-existent vectors get the largest relative distance of 100%.
            let (a, b) = SHARE_OF_MAX_DISTANCE;
            (RingElement(a), RingElement(b))
        };
        additive_shares.push(code_dist);
        // When applying the trick dot on trimmed masks, we have to multiply with 2 the
        // result The intuition being that a GaloisRingTrimmedMask contains half
        // the elements that a full GaloisRingMask has.
        additive_shares.push(mask_dist);
    }

    let batch_size = count as f64;
    let duration = start.elapsed().as_secs_f64() / batch_size;
    PAIRWISE_DISTANCE_METRICS.with_borrow_mut(|[metric_batch_size, metric_per_pair_duration]| {
        metric_batch_size.record(batch_size);
        metric_per_pair_duration.record(duration);
    });

    additive_shares
}

/// This is similar to `pairwise_distance`, but performs dot products on all rotations of the query.
pub fn rotation_aware_pairwise_distance<'a, I>(
    query: &'a ArcIris,
    targets: I,
) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    let start = Instant::now();
    let mut count = 0;
    let mut additive_shares = Vec::with_capacity(2 * ROTATIONS * targets.len());

    for target in targets {
        for rotation in IrisRotation::all() {
            let (code_dist, mask_dist) = if let Some(y) = target {
                count += 1;
                let (a, b) = (
                    query.code.rotation_aware_trick_dot(&y.code, &rotation),
                    query.mask.rotation_aware_trick_dot(&y.mask, &rotation),
                );
                (RingElement(a), RingElement(2) * RingElement(b))
            } else {
                // Non-existent vectors get the largest relative distance of 100%.
                let (a, b) = SHARE_OF_MAX_DISTANCE;
                (RingElement(a), RingElement(b))
            };
            additive_shares.push(code_dist);
            additive_shares.push(mask_dist);
        }
    }

    let batch_size = count as f64;
    let duration = start.elapsed().as_secs_f64() / batch_size;
    PAIRWISE_DISTANCE_METRICS.with_borrow_mut(|[metric_batch_size, metric_per_pair_duration]| {
        metric_batch_size.record(batch_size);
        metric_per_pair_duration.record(duration);
    });
    additive_shares
}

pub fn non_existent_distance() -> Vec<RingElement<u16>> {
    vec![
        RingElement(SHARE_OF_MAX_DISTANCE.0),
        RingElement(SHARE_OF_MAX_DISTANCE.1),
    ]
}

/// Converts additive sharing (from trick_dot output) to a replicated sharing by
/// masking it with a zero sharing
pub async fn galois_ring_to_rep3(
    session: &mut Session,
    items: Vec<RingElement<u16>>,
) -> Result<Vec<Share<u16>>> {
    let network = &mut session.network_session;

    // make sure we mask the input with a zero sharing
    let masked_items: Vec<_> = items
        .iter()
        .map(|x| session.prf.gen_zero_share() + x)
        .collect();

    // sending to the next party
    network
        .send_next(NetworkValue::VecRing16(masked_items.clone()))
        .await?;

    // receiving from previous party
    let shares_b = {
        match network.receive_prev().await {
            Ok(NetworkValue::VecRing16(message)) => Ok(message),
            _ => Err(eyre!("Error in receiving in galois_ring_to_rep3 operation")),
        }
    }?;
    let res: Vec<Share<u16>> = masked_items
        .into_iter()
        .zip(shares_b)
        .map(|(a, b)| Share::new(a, b))
        .collect();
    Ok(res)
}

/// Compares the given distance to a threshold and reveal the bit "less than or equal".
pub async fn lte_threshold_and_open(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
) -> Result<Vec<bool>> {
    let bits = greater_than_threshold(session, distances).await?;
    open_bin(session, &bits)
        .await
        .map(|v| v.into_iter().map(|x| x.convert().not()).collect())
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn open_ring<T: IntRing2k + NetworkInt>(
    session: &mut Session,
    shares: &[Share<T>],
) -> Result<Vec<T>> {
    let network = &mut session.network_session;
    let message = if shares.len() == 1 {
        T::new_network_element(shares[0].b)
    } else {
        let shares = shares.iter().map(|x| x.b).collect::<Vec<_>>();
        T::new_network_vec(shares)
    };

    network.send_next(message).await?;

    // receiving from previous party
    let c = network
        .receive_prev()
        .await
        .and_then(|v| T::into_vec(v))
        .map_err(|e| eyre!("Error in receiving in open operation: {}", e))?;

    // ADD shares with the received shares
    izip!(shares.iter(), c.iter())
        .map(|(s, c)| Ok((s.a + s.b + c).convert()))
        .collect::<Result<Vec<_>>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::local::{generate_local_identities, LocalRuntime},
        network::value::NetworkInt,
        protocol::{ops::NetworkValue::RingElement32, shared_iris::GaloisRingSharedIris},
        shares::{int_ring::IntRing2k, ring_impl::RingElement},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::{Rng, RngCore, SeedableRng};
    use rand_distr::{Distribution, Standard};
    use rstest::rstest;
    use std::{array, collections::HashMap, sync::Arc};
    use tokio::{sync::Mutex, task::JoinSet};
    use tracing::trace;

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_single(session: &mut Session, x: Share<u32>) -> Result<RingElement<u32>> {
        let network = &mut session.network_session;
        network.send_next(RingElement32(x.b)).await?;
        let missing_share = match network.receive_prev().await {
            Ok(NetworkValue::RingElement32(element)) => element,
            _ => bail!("Could not deserialize RingElement32"),
        };
        let (a, b) = x.get_ab();
        Ok(a + b + missing_share)
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_t_many<T>(session: &mut Session, shares: Vec<Share<T>>) -> Result<Vec<T>>
    where
        T: IntRing2k + NetworkInt,
    {
        let network = &mut session.network_session;

        let shares_b: Vec<_> = shares.iter().map(|s| s.b).collect();
        let message = shares_b;
        network.send_next(T::new_network_vec(message)).await?;

        // receiving from previous party
        let shares_c = {
            let net_message = network.receive_prev().await?;
            T::into_vec(net_message)
        }?;

        let res = shares
            .into_iter()
            .zip(shares_c)
            .map(|(s, c)| {
                let (a, b) = s.get_ab();
                (a + b + c).convert()
            })
            .collect();
        Ok(res)
    }

    #[tokio::test]
    async fn test_async_prf_setup() {
        let num_parties = 3;
        let identities = generate_local_identities();
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let mut runtime = LocalRuntime::new(identities.clone(), seeds.clone())
            .await
            .unwrap();

        // check whether parties have sent/received the correct seeds.
        // P0: [seed_0, seed_2]
        // P1: [seed_1, seed_0]
        // P2: [seed_2, seed_1]
        // This is done by calling next() on the PRFs and see whether they match with
        // the ones created from scratch.

        // Alice
        let prf0 = &mut runtime.sessions[0].prf;
        assert_eq!(
            prf0.get_my_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf0.get_prev_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_prev_prf().next_u64()
        );

        // Bob
        let prf1 = &mut runtime.sessions[1].prf;
        assert_eq!(
            prf1.get_my_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf1.get_prev_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_prev_prf().next_u64()
        );

        // Charlie
        let prf2 = &mut runtime.sessions[2].prf;
        assert_eq!(
            prf2.get_my_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf2.get_prev_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_prev_prf().next_u64()
        );
    }

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
    struct LocalShares1D<T: IntRing2k> {
        p0: Vec<Share<T>>,
        p1: Vec<Share<T>>,
        p2: Vec<Share<T>>,
    }

    fn create_array_sharing<R: RngCore, T: IntRing2k>(
        rng: &mut R,
        input: &Vec<T>,
    ) -> LocalShares1D<T>
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
        LocalShares1D {
            p0: player0,
            p1: player1,
            p2: player2,
        }
    }

    #[tokio::test]
    async fn test_replicated_cross_mul_lift() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let four_items = vec![1, 2, 3, 4];

        let four_shares = create_array_sharing(&mut rng, &four_items);

        let num_parties = 3;
        let identities = generate_local_identities();

        let four_share_map = HashMap::from([
            (identities[0].clone(), four_shares.p0),
            (identities[1].clone(), four_shares.p1),
            (identities[2].clone(), four_shares.p2),
        ]);

        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let runtime = LocalRuntime::new(identities.clone(), seeds.clone())
            .await
            .unwrap();

        let sessions: Vec<Arc<Mutex<Session>>> = runtime
            .sessions
            .into_iter()
            .map(|s| Arc::new(Mutex::new(s)))
            .collect();

        let mut jobs = JoinSet::new();
        for session in sessions {
            let session_lock = session.lock().await;
            let four_shares = four_share_map
                .get(&session_lock.own_identity())
                .unwrap()
                .clone();
            let session = session.clone();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                let four_shares = batch_signed_lift_vec(&mut session, four_shares)
                    .await
                    .unwrap();
                let out_shared = cross_mul(
                    &mut session,
                    &[(
                        DistanceShare {
                            code_dot: four_shares[0].clone(),
                            mask_dot: four_shares[1].clone(),
                        },
                        DistanceShare {
                            code_dot: four_shares[2].clone(),
                            mask_dot: four_shares[3].clone(),
                        },
                    )],
                )
                .await
                .unwrap()[0]
                    .clone();

                open_single(&mut session, out_shared).await.unwrap()
            });
        }
        // check first party output is equal to the expected result.
        let t = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(t, RingElement(2));
    }

    #[tokio::test]
    async fn test_compare_threshold_buckets() {
        const NUM_BUCKETS: usize = 100;
        const NUM_ITEMS: usize = 20;
        let mut rng = AesRng::seed_from_u64(0_u64);
        let items = (0..NUM_ITEMS)
            .flat_map(|_| {
                let mask = rng.gen_range(6000u32..12000);
                let code = rng.gen_range(-12000i16..12000);
                [code as u16 as u32, mask]
            })
            .collect_vec();

        let shares = create_array_sharing(&mut rng, &items);

        let thresholds: [f64; NUM_BUCKETS] =
            array::from_fn(|i| i as f64 / (NUM_BUCKETS * 2) as f64);
        let threshold_a_terms = thresholds
            .iter()
            .map(|x| translate_threshold_a(*x))
            .collect_vec();

        let num_parties = 3;
        let identities = generate_local_identities();

        let share_map = HashMap::from([
            (identities[0].clone(), shares.p0),
            (identities[1].clone(), shares.p1),
            (identities[2].clone(), shares.p2),
        ]);

        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let runtime = LocalRuntime::new(identities.clone(), seeds.clone())
            .await
            .unwrap();

        let sessions: Vec<Arc<Mutex<Session>>> = runtime
            .sessions
            .into_iter()
            .map(|s| Arc::new(Mutex::new(s)))
            .collect();

        let mut jobs = JoinSet::new();
        for session in sessions {
            let session_lock = session.lock().await;
            let shares = share_map.get(&session_lock.own_identity()).unwrap().clone();
            let session = session.clone();
            let threshold_a_terms = threshold_a_terms.clone();
            jobs.spawn(async move {
                let mut session = session.lock().await;
                let distances = shares[..]
                    .chunks_exact(2)
                    .map(|x| DistanceShare {
                        code_dot: x[0].clone(),
                        mask_dot: x[1].clone(),
                    })
                    .collect_vec();

                let bucket_result_shares =
                    compare_threshold_buckets(&mut session, &threshold_a_terms, &distances)
                        .await
                        .unwrap();

                open_ring(&mut session, &bucket_result_shares)
                    .await
                    .unwrap()
            });
        }
        // check first party output is equal to the expected result.
        let t1 = jobs.join_next().await.unwrap().unwrap();
        let t2 = jobs.join_next().await.unwrap().unwrap();
        let t3 = jobs.join_next().await.unwrap().unwrap();
        let expected = items[..]
            .chunks_exact(2)
            .fold([0; NUM_BUCKETS], |mut acc, x| {
                let code_dist = x[0];
                let mask_dist = x[1];
                for (i, &threshold) in thresholds.iter().enumerate() {
                    let threshold_a = translate_threshold_a(threshold);
                    let diff = (mask_dist * threshold_a).wrapping_sub(code_dist * 2u32.pow(16));
                    acc[i] += if (diff as i32) < 0 { 1 } else { 0 };
                }
                acc
            });
        assert_eq!(t1, expected);
        assert_eq!(t2, expected);
        assert_eq!(t3, expected);
    }

    #[tokio::test]
    async fn test_compare_min_threshold_buckets() {
        const NUM_BUCKETS: usize = 100;
        const NUM_ITEMS: usize = 20;
        const MAX_TEST_ROTATIONS: usize = 15;
        let mut rng = AesRng::seed_from_u64(0_u64);
        let sizes = (0..NUM_ITEMS)
            .map(|_| rng.gen_range(1..=MAX_TEST_ROTATIONS))
            .collect_vec();
        let flat_size = sizes.iter().sum::<usize>();

        let items = (0..flat_size)
            .flat_map(|_| {
                let mask = rng.gen_range(6000u32..12000);
                let code = rng.gen_range(-12000i16..12000);
                [code as u16 as u32, mask]
            })
            .collect_vec();

        let shares = create_array_sharing(&mut rng, &items);

        let thresholds: [f64; NUM_BUCKETS] =
            array::from_fn(|i| i as f64 / (NUM_BUCKETS * 2) as f64);
        let threshold_a_terms = thresholds
            .iter()
            .map(|x| translate_threshold_a(*x))
            .collect_vec();

        let num_parties = 3;
        let identities = generate_local_identities();

        let share_map = HashMap::from([
            (identities[0].clone(), shares.p0),
            (identities[1].clone(), shares.p1),
            (identities[2].clone(), shares.p2),
        ]);

        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let runtime = LocalRuntime::new(identities.clone(), seeds.clone())
            .await
            .unwrap();

        let sessions: Vec<Arc<Mutex<Session>>> = runtime
            .sessions
            .into_iter()
            .map(|s| Arc::new(Mutex::new(s)))
            .collect();

        let mut jobs = JoinSet::new();
        for session in sessions {
            let session_lock = session.lock().await;
            let shares = share_map.get(&session_lock.own_identity()).unwrap().clone();
            let session = session.clone();
            let threshold_a_terms = threshold_a_terms.clone();
            jobs.spawn({
                let sizes = sizes.clone();
                async move {
                    let mut session = session.lock().await;
                    let mut counter = 0;
                    let grouped_distances = sizes
                        .iter()
                        .map(|&size| {
                            (0..size)
                                .map(|_| {
                                    let code_dot = shares[counter].clone();
                                    let mask_dot = shares[counter + 1].clone();
                                    counter += 2;
                                    DistanceShare { code_dot, mask_dot }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();

                    let bucket_result_shares = compare_min_threshold_buckets(
                        &mut session,
                        &threshold_a_terms,
                        &grouped_distances,
                    )
                    .await
                    .unwrap();

                    open_ring(&mut session, &bucket_result_shares)
                        .await
                        .unwrap()
                }
            });
        }
        // check first party output is equal to the expected result.
        let t1 = jobs.join_next().await.unwrap().unwrap();
        let t2 = jobs.join_next().await.unwrap().unwrap();
        let t3 = jobs.join_next().await.unwrap().unwrap();

        let mut counter = 0;
        let grouped_distances = sizes
            .iter()
            .map(|&size| {
                (0..size)
                    .map(|_| {
                        let code_dist = items[counter];
                        let mask_dist = items[counter + 1];
                        counter += 2;
                        (code_dist, mask_dist)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let expected = grouped_distances
            .iter()
            .map(|group| {
                // reduce distances in each group
                group
                    .iter()
                    .reduce(|a, b| {
                        // plain distance formula is (0.5 - code/2*mask), the below is that multiplied by 2
                        if (1f64 - a.0 as f64 / a.1 as f64) < (1f64 - b.0 as f64 / b.1 as f64) {
                            a
                        } else {
                            b
                        }
                    })
                    .expect("Expected at least one distance in the group")
            })
            .fold([0; NUM_BUCKETS], |mut acc, x| {
                let code_dist = x.0;
                let mask_dist = x.1;
                for (i, &threshold) in thresholds.iter().enumerate() {
                    let threshold_a = translate_threshold_a(threshold);
                    let diff = (mask_dist * threshold_a).wrapping_sub(code_dist * 2u32.pow(16));
                    acc[i] += if (diff as i32) < 0 { 1 } else { 0 };
                }
                acc
            });
        assert_eq!(t1, expected);
        assert_eq!(t2, expected);
        assert_eq!(t3, expected);
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_additive(session: &mut Session, x: Vec<RingElement<u16>>) -> Result<Vec<u16>> {
        let prev_role = session.prev_identity()?;
        let network = &mut session.network_session;

        network
            .send_next(NetworkValue::VecRing16(x.clone()))
            .await?;

        let message_bytes = NetworkValue::VecRing16(x.clone());
        trace!(target: "searcher::network", action = "send", party = ?prev_role, bytes = x.len() * size_of::<u16>(), rounds = 0);

        network.send_prev(message_bytes).await?;

        let reply_0 = network.receive_prev().await;
        let reply_1 = network.receive_next().await;

        let missing_share_0 = match reply_0 {
            Ok(NetworkValue::VecRing16(element)) => element,
            _ => bail!("Could not deserialize VecRingElement16"),
        };
        let missing_share_1 = match reply_1 {
            Ok(NetworkValue::VecRing16(element)) => element,
            _ => bail!("Could not deserialize VecRingElement16"),
        };
        let opened_value: Vec<u16> = x
            .iter()
            .enumerate()
            .map(|(i, v)| (missing_share_0[i] + missing_share_1[i] + v).convert())
            .collect();
        Ok(opened_value)
    }

    #[tokio::test]
    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    async fn test_galois_ring_to_rep3(#[case] seed: u64) {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut rng = AesRng::seed_from_u64(seed);

        let iris_db = IrisDB::new_random_rng(2, &mut rng).db;

        let first_entry =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let second_entry =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[1].clone());

        let mut jobs = JoinSet::new();
        for (index, session) in sessions.iter().enumerate() {
            let own_shares = vec![(first_entry[index].clone(), second_entry[index].clone())]
                .into_iter()
                .map(|(x, mut y)| {
                    y.code.preprocess_iris_code_query_share();
                    y.mask.preprocess_mask_code_query_share();
                    Some((Arc::new(x), Arc::new(y)))
                })
                .collect_vec();
            let session = session.clone();
            jobs.spawn(async move {
                let mut player_session = session.lock().await;
                let x = galois_ring_pairwise_distance(own_shares);
                let opened_x = open_additive(&mut player_session, x.clone()).await.unwrap();
                let x_rep = galois_ring_to_rep3(&mut player_session, x).await.unwrap();
                let opened_x_rep = open_t_many(&mut player_session, x_rep).await.unwrap();
                (opened_x, opened_x_rep)
            });
        }
        let output0 = jobs.join_next().await.unwrap().unwrap();
        let output1 = jobs.join_next().await.unwrap().unwrap();
        let output2 = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(output0, output1);
        assert_eq!(output0, output2);

        let (plain_d1, plain_d2) = iris_db[0].get_dot_distance_fraction(&iris_db[1]);
        assert_eq!(output0.0[0], plain_d1 as u16);
        assert_eq!(output0.0[1], plain_d2);

        assert_eq!(output0.1[0], plain_d1 as u16);
        assert_eq!(output0.1[1], plain_d2);
    }
}
