use crate::{
    execution::session::{Session, SessionHandles},
    protocol::shared_iris::ArcIris,
};
use ampc_actor_utils::fast_metrics::FastHistogram;
pub use ampc_actor_utils::protocol::ops::{
    galois_ring_to_rep3, lt_zero_and_open_u16, open_ring, setup_replicated_prf, setup_shared_seed,
    sub_pub,
};
use ampc_actor_utils::protocol::{
    binary::{
        and_product, bit_inject, extract_msb_batch, lift, mul_lift_2k, open_bin, single_extract_msb,
    },
    ops::{conditionally_select_distance, cross_mul, oblivious_cross_compare},
};
pub use ampc_secret_sharing::shares::{
    bit::Bit,
    ring_impl::{RingElement, VecRingElement},
    share::{reconstruct_distance_vector, DistanceShare, Share},
    vecshare::VecShare,
};
use eyre::{bail, eyre, Result};
use iris_mpc_common::{
    galois_engine::degree4::{IrisRotation, SHARE_OF_MAX_DISTANCE},
    ROTATIONS,
};
use itertools::{izip, Itertools};
use std::{cmp::Ordering, ops::Not, time::Instant};
use tracing::instrument;

pub(crate) const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub(crate) const B_BITS: u64 = 16;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;

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
            let x = d.mask_dot * A as u32;
            let y = d.code_dot * B as u32;
            y - x
        })
        .collect();

    extract_msb_batch(session, &diffs).await
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

    single_extract_msb(session, y).await
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
                let diff = *left - *right;
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
        let mut res = c * RingElement(diff);
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
        .flat_map(|(id, d)| [id, d.code_dot, d.mask_dot])
        .collect_vec();
    let right_dist = right_distances
        .into_iter()
        .flat_map(|(id, d)| [id, d.code_dot, d.mask_dot])
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
            (shared_index, *d)
        })
        .collect_vec();
    // Lift swap bits to u32 shares
    let swap_bits_u32 = bit_inject(session, VecShare::<Bit>::new_vec(swap_bits))
        .await?
        .inner();

    let distances_to_swap = indices
        .iter()
        .filter_map(|(idx1, idx2)| match (list.get(*idx1), list.get(*idx2)) {
            (Some((_, d1)), Some((_, d2))) => Some((*d1, *d2)),
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
                d_pair.0.code_dot + d_pair.1.code_dot - first_d.code_dot,
                d_pair.0.mask_dot + d_pair.1.mask_dot - first_d.mask_dot,
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
        let first_id = bit * id1 + not_bit * id2;
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
    let swap_bits_u32 = bit_inject(session, VecShare::<Bit>::new_vec(swap_bits))
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

            let id = mul_share_a(*id1, *id2, sb);
            let code_dot_a = mul_share_a(d1.code_dot, d2.code_dot, sb);
            let mask_dot_a = mul_share_a(d1.mask_dot, d2.mask_dot, sb);
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
            let first_id = res_id + *id2;
            let second_id = *id1 - res_id;
            let first_distance = DistanceShare {
                code_dot: res_dist.code_dot + dist2.code_dot,
                mask_dot: res_dist.mask_dot + dist2.mask_dot,
            };
            let second_distance = DistanceShare {
                code_dot: dist1.code_dot - res_dist.code_dot,
                mask_dot: dist1.mask_dot - res_dist.mask_dot,
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

// consider putting this in ampc-common next to oblivious_cross_compare()
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
    let bits = extract_msb_batch(session, &diff).await?;
    // Open the MSB
    let opened_b = open_bin(session, &bits).await?;
    opened_b.into_iter().map(|x| Ok(x.convert())).collect()
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
pub(crate) async fn min_round_robin_batch(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
    batch_size: usize,
) -> Result<Vec<DistanceShare<u32>>> {
    if distances.is_empty() {
        eyre::bail!("Expected at least one distance share");
    }
    if distances.len() % batch_size != 0 {
        eyre::bail!("Distances length must be a multiple of batch size");
    }
    if batch_size < 2 {
        return Ok(distances.to_vec());
    }

    // Within each batch, compute all the pairwise comparisons in a round-robin fashion.
    // The resulting comparison table looks like
    //
    //    | d0 | d1 | d2 | d3 |
    // ------------------------
    // d0 | -  | b01| b02| b03|
    // d1 |    | -  | b12| b13|
    // d2 |    |    | -  | b23|
    // d3 |    |    |    | -  |
    //
    // where `bij` is the bit corresponding to `di < dj`.
    // Comparison bits are arranged in a flat vector as
    // `[b01, b02, b03, b12, b13, b23]`.
    let num_batches = distances.len() / batch_size;
    let mut pairs = Vec::with_capacity(num_batches * (batch_size * (batch_size - 1) / 2));
    for i_batch in 0..num_batches {
        for i in 0..batch_size {
            for j in (i + 1)..batch_size {
                let distance_i = distances[i * num_batches + i_batch];
                let distance_j = distances[j * num_batches + i_batch];
                pairs.push((distance_i, distance_j));
            }
        }
    }
    let comparison_bits = oblivious_cross_compare(session, &pairs).await?;
    // Fill in the rest of the comparison table by setting diagonal bits to 1 and negating the bits above the diagonal.
    // In other words, the `[i][j]`-th value of the table is equal to the bit
    // - `di < dj` if `i < j`, or
    // - `di <= dj` if `i >= j`.
    //
    //    | d0 | d1 | d2 | d3 |
    // ------------------------
    // d0 | 1  | b01| b02| b03|
    // d1 |!b01| 1  | b12| b13|
    // d2 |!b02|!b12| 1  | b23|
    // d3 |!b03|!b13|!b23| 1  |
    //
    // Extract this table column-wise as `batch_matrix` to AND them element-wise.
    // Group jth columns together, i.e., return a matrix `batch_selection_bits`, where `batch_selection_bits[j]` contains the comparison bits
    // between distance `j` of every batch and all the other distances within the same batch.
    let mut batch_selection_bits = (0..batch_size)
        .map(|_| VecShare::with_capacity(num_batches * batch_size))
        .collect_vec();
    for batch in comparison_bits.chunks(batch_size * (batch_size - 1) / 2) {
        let mut batch_matrix = (0..batch_size)
            .map(|_| VecShare::with_capacity(batch_size))
            .collect_vec();
        let mut batch_counter = 0;
        for i in 0..batch_size {
            for j in 0..batch_size {
                let value = match i.cmp(&j) {
                    Ordering::Less => {
                        batch_counter += 1;
                        batch[batch_counter - 1]
                    }
                    Ordering::Equal => Share::from_const(Bit::new(true), session.own_role()),
                    Ordering::Greater => batch_matrix[i].get_at(j).not(),
                };
                batch_matrix[j].push(value);
            }
        }
        for (j, column_bits) in batch_matrix.into_iter().enumerate() {
            batch_selection_bits[j].extend(column_bits);
        }
    }
    // Compute the AND of each row in the `batch_selection_bits` matrix.
    // This gives us, for each distance in the batch, whether it is the minimum distance in its batch.
    let selection_bits =
        and_product(session, batch_selection_bits, num_batches * batch_size).await?;
    // The resulting bits are bit injected into u32.
    let selection_bits: VecShare<u32> = bit_inject(session, selection_bits).await?;
    // Multiply distance shares with selection bits to zero out non-minimum distances.
    let selected_distances = {
        let mut shares_a = VecRingElement::with_capacity(2 * distances.len());
        for i_batch in 0..num_batches {
            for i in 0..batch_size {
                let distance = &distances[i * num_batches + i_batch];
                let b = &selection_bits.shares()[i_batch * batch_size + i];
                let code_a = session.prf.gen_zero_share() + b * &distance.code_dot;
                let mask_a = session.prf.gen_zero_share() + b * &distance.mask_dot;
                shares_a.push(code_a);
                shares_a.push(mask_a);
            }
        }

        let network = &mut session.network_session;
        network.send_ring_vec_next(&shares_a).await?;
        let shares_b = network.receive_ring_vec_prev().await?;
        reconstruct_distance_vector(shares_a, shares_b)
    };
    // Now sum up the selected distances within each batch.
    // Only one distance per batch is non-zero, so this gives us the minimum distance per batch.
    let res = selected_distances
        .chunks(batch_size)
        .map(|chunk| chunk.iter().cloned().reduce(|acc, a| acc + a).unwrap())
        .collect_vec();
    Ok(res)
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

/// Row-major prerotated query for L1 cache efficiency.
/// Layout: [row0_rot0..rot30][row1_rot0..rot30]...
pub struct PrerotatedQueryRowMajor {
    /// code: 16 rows × 31 rotations × 800 elements = 396,800 u16s (~793KB)
    code_data: Vec<u16>,
    /// mask: 8 rows × 31 rotations × 800 elements = 198,400 u16s (~397KB)
    mask_data: Vec<u16>,
}

impl PrerotatedQueryRowMajor {
    pub const ROW_SIZE: usize = 800;
    pub const CODE_ROWS: usize = 16;
    pub const MASK_ROWS: usize = 8;
    const CODE_SIZE: usize = Self::CODE_ROWS * ROTATIONS * Self::ROW_SIZE;
    const MASK_SIZE: usize = Self::MASK_ROWS * ROTATIONS * Self::ROW_SIZE;

    /// Rotation amounts for each of the 31 rotations.
    const ROTATION_AMOUNTS: [usize; ROTATIONS] = {
        let mut amounts = [0usize; ROTATIONS];
        // Left rotations: rotate left by 60, 56, ..., 4 elements
        let mut i = 0;
        while i < 15 {
            amounts[i] = (15 - i) * 4;
            i += 1;
        }
        // Center: no rotation
        amounts[15] = 0;
        // Right rotations: rotate right by 4, 8, ..., 60 (i.e., left by 796, 792, ..., 740)
        let mut i = 1;
        while i <= 15 {
            amounts[15 + i] = 800 - i * 4;
            i += 1;
        }
        amounts
    };

    /// Rotate row directly into destination buffer (zero allocations).
    #[inline]
    fn rotate_row_into(src: &[u16], dst: &mut [u16], left_amount: usize) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        debug_assert!(left_amount < len);
        // Rotate left by `left_amount`: [left_amount..] ++ [..left_amount]
        let (first, second) = src.split_at(left_amount);
        dst[..second.len()].copy_from_slice(second);
        dst[second.len()..].copy_from_slice(first);
    }

    /// Create a new buffer (allocates memory).
    fn new_buffer() -> Self {
        Self {
            code_data: vec![0u16; Self::CODE_SIZE],
            mask_data: vec![0u16; Self::MASK_SIZE],
        }
    }

    /// Fill the buffer with rotated query data (reuses existing allocation).
    fn fill(&mut self, query: &ArcIris) {
        // Process code rows - write directly into destination (no intermediate allocations)
        for row_idx in 0..Self::CODE_ROWS {
            let src_row =
                &query.code.coefs[row_idx * Self::ROW_SIZE..(row_idx + 1) * Self::ROW_SIZE];
            for rot_idx in 0..ROTATIONS {
                let dst_offset = (row_idx * ROTATIONS + rot_idx) * Self::ROW_SIZE;
                let dst = &mut self.code_data[dst_offset..dst_offset + Self::ROW_SIZE];
                Self::rotate_row_into(src_row, dst, Self::ROTATION_AMOUNTS[rot_idx]);
            }
        }

        // Process mask rows - write directly into destination (no intermediate allocations)
        for row_idx in 0..Self::MASK_ROWS {
            let src_row =
                &query.mask.coefs[row_idx * Self::ROW_SIZE..(row_idx + 1) * Self::ROW_SIZE];
            for rot_idx in 0..ROTATIONS {
                let dst_offset = (row_idx * ROTATIONS + rot_idx) * Self::ROW_SIZE;
                let dst = &mut self.mask_data[dst_offset..dst_offset + Self::ROW_SIZE];
                Self::rotate_row_into(src_row, dst, Self::ROTATION_AMOUNTS[rot_idx]);
            }
        }
    }

    pub fn new(query: &ArcIris) -> Self {
        let mut buf = Self::new_buffer();
        buf.fill(query);
        buf
    }

    /// Get all 31 rotations of a code row (contiguous in memory - 50KB)
    #[inline]
    fn code_row_rotations(&self, row_idx: usize) -> &[u16] {
        let start = row_idx * ROTATIONS * Self::ROW_SIZE;
        let end = start + ROTATIONS * Self::ROW_SIZE;
        &self.code_data[start..end]
    }

    /// Get all 31 rotations of a mask row (contiguous in memory - 50KB)
    #[inline]
    fn mask_row_rotations(&self, row_idx: usize) -> &[u16] {
        let start = row_idx * ROTATIONS * Self::ROW_SIZE;
        let end = start + ROTATIONS * Self::ROW_SIZE;
        &self.mask_data[start..end]
    }
}

// Thread-local storage for reusable prerotated query buffers
thread_local! {
    static PREROTATED_BUFFER: std::cell::RefCell<Option<PrerotatedQueryRowMajor>> = const { std::cell::RefCell::new(None) };
}

/// Row-major rotation-aware distance - processes row-by-row for L1 cache efficiency.
/// Uses thread-local buffer reuse to minimize allocations at high thread counts.
pub fn rotation_aware_pairwise_distance_rowmajor<'a, I>(
    query: &'a ArcIris,
    targets: I,
) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    let target_count = targets.len();
    let mut additive_shares = vec![RingElement(0u16); 2 * ROTATIONS * target_count];

    // Use thread-local buffer to avoid allocation per call
    PREROTATED_BUFFER.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        let prerotated = borrowed.get_or_insert_with(PrerotatedQueryRowMajor::new_buffer);
        prerotated.fill(query);
        rotation_aware_inner(prerotated, targets, &mut additive_shares);
    });

    additive_shares
}

/// Inner implementation that works with a borrowed prerotated buffer.
#[inline(never)]
fn rotation_aware_inner<'a, I>(
    prerotated: &PrerotatedQueryRowMajor,
    targets: I,
    additive_shares: &mut [RingElement<u16>],
) where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    const CODE_ROWS: usize = PrerotatedQueryRowMajor::CODE_ROWS;
    const MASK_ROWS: usize = PrerotatedQueryRowMajor::MASK_ROWS;

    let targets: Vec<_> = targets.collect();

    // Process row-by-row: all 31 rotations of one row stay in L1
    for row_idx in 0..CODE_ROWS {
        let query_rows = prerotated.code_row_rotations(row_idx); // 50KB, fits in L1

        for (target_idx, target_opt) in targets.iter().enumerate() {
            if let Some(target) = target_opt {
                let target_row = &target.code.coefs[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];

                for rot_idx in 0..ROTATIONS {
                    let query_row = &query_rows[rot_idx * ROW_SIZE..(rot_idx + 1) * ROW_SIZE];
                    let partial = simple_dot_product(query_row, target_row);
                    let result_idx = target_idx * ROTATIONS * 2 + rot_idx * 2;
                    additive_shares[result_idx] =
                        RingElement(additive_shares[result_idx].0.wrapping_add(partial));
                }
            }
        }
    }

    // Process mask rows - accumulate first, multiply by 2 later
    for row_idx in 0..MASK_ROWS {
        let query_rows = prerotated.mask_row_rotations(row_idx);

        for (target_idx, target_opt) in targets.iter().enumerate() {
            if let Some(target) = target_opt {
                let target_row = &target.mask.coefs[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];

                for rot_idx in 0..ROTATIONS {
                    let query_row = &query_rows[rot_idx * ROW_SIZE..(rot_idx + 1) * ROW_SIZE];
                    let partial = simple_dot_product(query_row, target_row);
                    let result_idx = target_idx * ROTATIONS * 2 + rot_idx * 2 + 1;
                    additive_shares[result_idx] =
                        RingElement(additive_shares[result_idx].0.wrapping_add(partial));
                }
            }
        }
    }

    // Multiply mask results by 2 for Some targets, set max distance for None targets
    for (target_idx, target_opt) in targets.iter().enumerate() {
        let base_idx = target_idx * ROTATIONS * 2;
        if target_opt.is_some() {
            for rot_idx in 0..ROTATIONS {
                let mask_idx = base_idx + rot_idx * 2 + 1;
                additive_shares[mask_idx] = RingElement(2) * additive_shares[mask_idx];
            }
        } else {
            let (a, b) = SHARE_OF_MAX_DISTANCE;
            for rot_idx in 0..ROTATIONS {
                let code_idx = base_idx + rot_idx * 2;
                additive_shares[code_idx] = RingElement(a);
                additive_shares[code_idx + 1] = RingElement(b);
            }
        }
    }
}

#[inline]
fn simple_dot_product(a: &[u16], b: &[u16]) -> u16 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0u16;
    for i in 0..a.len() {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
    }
    sum
}

pub fn non_existent_distance() -> Vec<RingElement<u16>> {
    vec![
        RingElement(SHARE_OF_MAX_DISTANCE.0),
        RingElement(SHARE_OF_MAX_DISTANCE.1),
    ]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::local::{generate_local_identities, LocalRuntime},
        network::value::{NetworkInt, NetworkValue},
        protocol::shared_iris::GaloisRingSharedIris,
        shares::{int_ring::IntRing2k, ring_impl::RingElement},
    };
    use aes_prng::AesRng;
    use ampc_actor_utils::protocol::{ops::batch_signed_lift_vec, prf::Prf};
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::{Rng, RngCore, SeedableRng};
    use rand_distr::{Distribution, Standard};
    use rstest::rstest;
    use std::{collections::HashMap, sync::Arc};
    use tokio::{sync::Mutex, task::JoinSet};
    use tracing::trace;
    use NetworkValue::RingElement32;

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
                            code_dot: four_shares[0],
                            mask_dot: four_shares[1],
                        },
                        DistanceShare {
                            code_dot: four_shares[2],
                            mask_dot: four_shares[3],
                        },
                    )],
                )
                .await
                .unwrap()[0];

                open_single(&mut session, out_shared).await.unwrap()
            });
        }
        // check first party output is equal to the expected result.
        let t = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(t, RingElement(2));
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

    #[rstest]
    #[case(0, 1)]
    #[case(1, 5)]
    #[case(42, 10)]
    fn test_rotation_aware_pairwise_distance_rowmajor_equivalence(
        #[case] seed: u64,
        #[case] num_targets: usize,
    ) {
        use crate::protocol::shared_iris::GaloisRingSharedIris;

        let mut rng = AesRng::seed_from_u64(seed);

        // Create enough iris codes: 1 for query + num_targets for targets
        let iris_db = IrisDB::new_random_rng(1 + num_targets, &mut rng).db;

        // Generate shares for the query
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());

        // Generate shares for targets
        let target_shares: Vec<_> = iris_db[1..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        // Test for each party's share
        for party_id in 0..3 {
            // Prepare query with preprocessing
            let mut query = query_shares[party_id].clone();
            query.code.preprocess_iris_code_query_share();
            query.mask.preprocess_mask_code_query_share();
            let query_arc = Arc::new(query);

            // Prepare targets (no preprocessing needed for targets)
            let targets: Vec<ArcIris> = target_shares
                .iter()
                .map(|shares| Arc::new(shares[party_id].clone()))
                .collect();

            // Call the original function
            let result_original =
                rotation_aware_pairwise_distance(&query_arc, targets.iter().map(Some));

            // Call the row-major function
            let result_rowmajor =
                rotation_aware_pairwise_distance_rowmajor(&query_arc, targets.iter().map(Some));

            // Verify results are identical
            assert_eq!(
                result_original.len(),
                result_rowmajor.len(),
                "Result lengths should match"
            );
            for (i, (orig, rowmaj)) in result_original
                .iter()
                .zip(result_rowmajor.iter())
                .enumerate()
            {
                assert_eq!(
                    orig, rowmaj,
                    "Mismatch at index {} for party {} with seed {}: original {:?} != rowmajor {:?}",
                    i, party_id, seed, orig, rowmaj
                );
            }
        }
    }

    #[test]
    fn test_rotation_aware_pairwise_distance_rowmajor_with_none_targets() {
        use crate::protocol::shared_iris::GaloisRingSharedIris;

        let mut rng = AesRng::seed_from_u64(123);

        // Create iris codes
        let iris_db = IrisDB::new_random_rng(3, &mut rng).db;

        // Generate shares
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let target_shares: Vec<_> = iris_db[1..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        // Test for party 0
        let party_id = 0;

        // Prepare query with preprocessing
        let mut query = query_shares[party_id].clone();
        query.code.preprocess_iris_code_query_share();
        query.mask.preprocess_mask_code_query_share();
        let query_arc = Arc::new(query);

        // Keep the Arc values alive
        let target0: ArcIris = Arc::new(target_shares[0][party_id].clone());
        let target1: ArcIris = Arc::new(target_shares[1][party_id].clone());
        let targets_with_none: Vec<Option<&ArcIris>> = vec![
            Some(&target0),
            None, // This should get SHARE_OF_MAX_DISTANCE
            Some(&target1),
        ];

        // Call both functions
        let result_original =
            rotation_aware_pairwise_distance(&query_arc, targets_with_none.clone().into_iter());
        let result_rowmajor =
            rotation_aware_pairwise_distance_rowmajor(&query_arc, targets_with_none.into_iter());

        // Verify results are identical
        assert_eq!(result_original.len(), result_rowmajor.len());
        for (i, (orig, rowmaj)) in result_original
            .iter()
            .zip(result_rowmajor.iter())
            .enumerate()
        {
            assert_eq!(orig, rowmaj, "Mismatch at index {}", i);
        }

        // Verify that None targets produce SHARE_OF_MAX_DISTANCE
        let (max_code, max_mask) = SHARE_OF_MAX_DISTANCE;
        // Second target is None, so indices 31*2..31*4 should all be max distance
        for rot in 0..ROTATIONS {
            let idx = ROTATIONS * 2 + rot * 2;
            assert_eq!(
                result_rowmajor[idx].0, max_code,
                "Code at rotation {} should be max",
                rot
            );
            assert_eq!(
                result_rowmajor[idx + 1].0,
                max_mask,
                "Mask at rotation {} should be max",
                rot
            );
        }
    }
}
