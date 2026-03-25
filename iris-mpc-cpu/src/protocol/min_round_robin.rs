use std::cmp::Ordering;
use std::ops::Not;

use ampc_actor_utils::{
    execution::session::{Session, SessionHandles},
    network::mpc::NetworkInt,
    protocol::binary::{and_product, bit_inject},
};
use ampc_secret_sharing::{
    shares::{bit::Bit, DistanceShare, RingRandFillable, VecRingElement, VecShare},
    IntRing2k, Share,
};
use eyre::Result;
use itertools::{izip, Itertools};
use rand_distr::{Distribution, Standard};

use crate::protocol::ops::DistancePair;

/// Builds round-robin comparison pairs from a flattened batch of distance shares.
///
/// Within each batch, compute all the pairs to be compared in a round-robin fashion,
/// namely, the pairs di, dj for all i < j, where di and dj are distances within the same batch.
fn build_round_robin_pairs<T: IntRing2k>(
    distances: &[DistanceShare<T>],
    batch_size: usize,
    num_batches: usize,
) -> Vec<DistancePair<T>> {
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
    pairs
}

/// Given round-robin comparison bits, selects the minimum distance in each batch.
///
/// This is the generic core shared by both FHD and NHD round-robin minimum.
async fn select_round_robin_min<T>(
    session: &mut Session,
    distances: &[DistanceShare<T>],
    batch_size: usize,
    num_batches: usize,
    comparison_bits: Vec<Share<Bit>>,
) -> Result<Vec<DistanceShare<T>>>
where
    T: IntRing2k + NetworkInt + RingRandFillable,
    Standard: Distribution<T>,
{
    // At the start, the comparison table looks like
    //
    //    | d0 | d1 | d2 | d3 |
    // ------------------------
    // d0 | -  | b01| b02| b03|
    // d1 |    | -  | b12| b13|
    // d2 |    |    | -  | b23|
    // d3 |    |    |    | -  |
    //
    // where `bij` is the bit corresponding to the distance comparison `di < dj`.
    // Comparison bits are arranged in a flat vector as
    // `[b01, b02, b03, b12, b13, b23]`.
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
    let selection_bits =
        and_product(session, batch_selection_bits, num_batches * batch_size).await?;
    // The resulting bits are bit injected into T.
    let selection_bits: VecShare<T> = bit_inject(session, selection_bits).await?;
    // Multiply distance shares with selection bits to zero out non-minimum distances.
    let selected_distances = {
        let mut shares_a: VecRingElement<T> = VecRingElement::with_capacity(2 * distances.len());
        for i_batch in 0..num_batches {
            for i in 0..batch_size {
                let distance = &distances[i * num_batches + i_batch];
                let b = &selection_bits.shares()[i_batch * batch_size + i];
                let code_a = session.prf.gen_zero_share::<T>() + b * &distance.code_dot;
                let mask_a = session.prf.gen_zero_share::<T>() + b * &distance.mask_dot;
                shares_a.push(code_a);
                shares_a.push(mask_a);
            }
        }

        let network = &mut session.network_session;
        network.send_ring_vec_next(&shares_a).await?;
        let shares_b: VecRingElement<T> = network.receive_ring_vec_prev().await?;
        // Reconstruct distance shares from (a, b) ring elements
        izip!(shares_a.0, shares_b.0)
            .map(|(a, b)| Share::new(a, b))
            .tuples()
            .map(|(code_dot, mask_dot)| DistanceShare::new(code_dot, mask_dot))
            .collect_vec()
    };
    // Now sum up the selected distances within each batch.
    // Only one distance per batch is non-zero, so this gives us the minimum distance per batch.
    let res = selected_distances
        .chunks(batch_size)
        .map(|chunk| chunk.iter().cloned().reduce(|acc, a| acc + a).unwrap())
        .collect_vec();
    Ok(res)
}

/// Generalization of cross-compare results for both FHD and NHD, boxed to avoid lifetime issues when passing into other functions.
pub(crate) type CrossCompareFnResult<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Share<Bit>>>> + Send + 'a>>;

/// Generalization of cross-compare function for both FHD and NHD, which can be passed into other functions.
pub(crate) type CrossCompareFn<T> =
    for<'a> fn(&'a mut Session, &'a [DistancePair<T>]) -> CrossCompareFnResult<'a>;

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
pub(crate) async fn min_round_robin_batch_with<T>(
    session: &mut Session,
    distances: &[DistanceShare<T>],
    batch_size: usize,
    cross_compare_fn: CrossCompareFn<T>,
) -> Result<Vec<DistanceShare<T>>>
where
    T: IntRing2k + NetworkInt + RingRandFillable,
    Standard: Distribution<T>,
{
    if distances.is_empty() {
        eyre::bail!("Expected at least one distance share");
    }
    if distances.len() % batch_size != 0 {
        eyre::bail!("Distances length must be a multiple of batch size");
    }
    if batch_size < 2 {
        return Ok(distances.to_vec());
    }
    let num_batches = distances.len() / batch_size;
    let pairs = build_round_robin_pairs(distances, batch_size, num_batches);
    let comparison_bits = cross_compare_fn(session, &pairs).await?;
    select_round_robin_min(session, distances, batch_size, num_batches, comparison_bits).await
}
