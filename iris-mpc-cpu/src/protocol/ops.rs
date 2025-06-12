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
        shared_iris::GaloisRingSharedIris,
    },
    shares::{
        bit::Bit,
        ring_impl::RingElement,
        share::{DistanceShare, Share},
        vecshare::VecShare,
        IntRing2k,
    },
};
use aes_prng::AesRng;
use eyre::{bail, eyre, Result};
use itertools::{izip, Itertools};
use rand::SeedableRng;
use std::array;
use std::collections::HashMap;
use tracing::instrument;
use tracing_subscriber::fmt;

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
    session
        .send_next(NetworkValue::PrfKey(my_seed).to_network())
        .await?;
    // received other seed from the previous party
    let serialized_other_seed = session.receive_prev().await;
    // deserializing received seed.
    let other_seed = match NetworkValue::from_network(serialized_other_seed) {
        Ok(NetworkValue::PrfKey(seed)) => seed,
        _ => bail!("Could not deserialize PrfKey"),
    };
    // creating the two PRFs
    Ok(Prf::new(my_seed, other_seed))
}

/// Setup an RNG common between all parties, for use in stochastic algorithms (e.g. HNSW layer selection).
pub async fn setup_shared_rng(session: &mut NetworkSession, my_seed: PrfSeed) -> Result<AesRng> {
    let my_msg = NetworkValue::PrfKey(my_seed).to_network();

    let decode = |msg| match NetworkValue::from_network(msg) {
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
    Ok(AesRng::from_seed(shared_seed))
}

/// Compares the distance between two iris pairs to a threshold.
///
/// - Takes as input two code and mask dot products between two irises,
///   i.e., code_dist = <iris1.code, iris2.code> and mask_dist = <iris1.mask, iris2.mask>.
/// - Lifts the two dot products to the ring Z_{2^32}.
/// - Multiplies with predefined threshold constants B = 2^16 and A = ((1. - 2.
///   * MATCH_THRESHOLD_RATIO) * B as f64).
/// - Compares mask_dist * A < code_dist * B.
pub async fn compare_threshold(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
) -> Result<Vec<Share<Bit>>> {
    let diffs: Vec<Share<u32>> = distances
        .iter()
        .map(|d| {
            let x = d.mask_dot.clone() * A as u32;
            let y = d.code_dot.clone() * B as u32;
            x - y
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

/// Converts distances from the format [((query_id, rotation_id), Vec<T>)]
/// to Vec<Vec<Vec<T>>> where outer Vec represents queries grouped by query_id,
/// middle Vec represents rotations ordered by rotation_id within each query,
/// and inner Vec represents distance shares for each rotation.
fn convert_distances_to_query_grouped<T: Clone>(
    distances: &[((u32, u32), Vec<T>)],
) -> Vec<Vec<Vec<T>>> {
    use std::collections::BTreeMap;

    // Group by query_id and collect into a BTreeMap to maintain order
    let mut query_map: BTreeMap<u32, BTreeMap<u32, Vec<T>>> = BTreeMap::new();

    for ((query_id, rotation_id), values) in distances {
        query_map
            .entry(*query_id)
            .or_default()
            .insert(*rotation_id, values.clone());
    }

    // Convert to the desired format: Vec<Vec<Vec<T>>>
    query_map
        .into_values()
        .map(|rotation_map| rotation_map.into_values().collect())
        .collect()
}

/// Compares the distance between two iris pairs to a list of thresholds, represented as t_i/B, with B = 2^16.
/// Use the [translate_threshold_a] function to compute the A term of the threshold comparison.
pub async fn compare_threshold_buckets(
    session: &mut Session,
    threshold_a_terms: &[u32],
    distances: &[((u32, u32), Vec<DistanceShare<u32>>)],
) -> Result<Vec<Share<u32>>> {
    let per_query_per_rotation_distances = convert_distances_to_query_grouped(distances);
    
    let per_query_multiplied_distances = per_query_per_rotation_distances.iter().enumerate().map(
        |(query_index, per_rotation_distance_shares)| {
            let per_rotation_shares = per_rotation_distance_shares
                .iter()
                .enumerate()
                .map(|(rotation_index, distance_shares)| {
                    let shares_per_rotation = threshold_a_terms
                        .iter()
                        .flat_map(|a| {
                            distance_shares.iter().map(|distance_share| {
                                let x = distance_share.mask_dot.clone() * *a;
                                let y = distance_share.code_dot.clone() * B as u32;
                                x - y
                            })
                        })
                        .collect_vec();
                    shares_per_rotation
                },
            ).collect_vec();
            
            per_rotation_shares.iter().map(async |per_rotation_shares| {
                tracing::info!("compare_threshold_buckets diffs length: {}", per_rotation_shares.len());
                let msbs = extract_msb_u32_batch(session, &per_rotation_shares).await;
                let msbs = VecShare::new_vec(msbs.unwrap());
                
                // bit_inject all MSBs into u32 to be able to add them up
                let sums = bit_inject_ot_2round(session, msbs).await;
            }).collect_vec();
            
            // perform multiplication on per_rotation_shares (code will come from dkales)
            let multiplied_per_rotation_shares: Vec<Share<u32>> = vec![];
            multiplied_per_rotation_shares
        },
    ).collect_vec();
    
    let buckets = per_query_multiplied_distances
        
        .into_iter()
        .chunks(distances.len())
        .into_iter()
        .map(|chunk| chunk.reduce(|a, b| a + b).unwrap_or_default())
        .collect_vec();

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
    let y = mul_lift_2k::<B_BITS>(&code_dist);
    let mut x = lift(session, VecShare::new_vec(vec![mask_dist])).await?;
    let mut x = x
        .pop()
        .ok_or(eyre!("Expected a single element in the VecShare"))?;
    x *= A as u32;
    x -= y;

    single_extract_msb_u32(session, x).await
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
    let res_a: Vec<RingElement<u32>> = distances
        .iter()
        .map(|(d1, d2)| {
            session.prf.gen_zero_share() + &d2.code_dot * &d1.mask_dot - &d1.code_dot * &d2.mask_dot
        })
        .collect();

    let network = &mut session.network_session;

    let message = if res_a.len() == 1 {
        NetworkValue::RingElement32(res_a[0]).to_network()
    } else {
        NetworkValue::VecRing32(res_a.clone()).to_network()
    };
    network.send_next(message).await?;

    let serialized_reply = network.receive_prev().await;
    let res_b = match NetworkValue::from_network(serialized_reply) {
        Ok(NetworkValue::RingElement32(element)) => vec![element],
        Ok(NetworkValue::VecRing32(elements)) => elements,
        _ => bail!("Could not deserialize RingElement32"),
    };
    Ok(izip!(res_a.into_iter(), res_b.into_iter())
        .map(|(a, b)| Share::new(a, b))
        .collect())
}

/// For every pair of distance shares (d1, d2), this computes the bit d2 < d1 and opens it.
///
/// The less-than operator is implemented in 2 steps:
///
/// 1. d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot is computed, which is a numerator of the fraction difference d2.code_dot / d2.mask_dot - d1.code_dot / d1.mask_dot.
/// 2. The most significant bit of the result is extracted.
///
/// Input values are assumed to be 16-bit shares that have been lifted to
/// 32-bit.
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

/// Computes the dot product between the iris pairs; for both the code and the
/// mask of the irises. We pack the dot products of the code and mask into one
/// vector to be able to reshare it later.
pub async fn galois_ring_pairwise_distance(
    _session: &mut Session,
    pairs: &[(&GaloisRingSharedIris, &GaloisRingSharedIris)],
) -> Vec<RingElement<u16>> {
    let mut additive_shares = Vec::with_capacity(2 * pairs.len());
    for pair in pairs.iter() {
        let (x, y) = pair;
        let code_dist = x.code.trick_dot(&y.code);
        let mask_dist = x.mask.trick_dot(&y.mask);
        additive_shares.push(RingElement(code_dist));
        // When applying the trick dot on trimmed masks, we have to multiply with 2 the
        // result The intuition being that a GaloisRingTrimmedMask contains half
        // the elements that a full GaloisRingMask has.
        additive_shares.push(RingElement(2) * RingElement(mask_dist));
    }
    additive_shares
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
        .send_next(NetworkValue::VecRing16(masked_items.clone()).to_network())
        .await?;

    // receiving from previous party

    let shares_b = {
        let serialized_other_share = network.receive_prev().await;
        match NetworkValue::from_network(serialized_other_share) {
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

/// Compares the given distance to a threshold and reveal the result.
pub async fn compare_threshold_and_open(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
) -> Result<Vec<bool>> {
    let bits = compare_threshold(session, distances).await?;
    open_bin(session, &bits)
        .await
        .map(|v| v.into_iter().map(|x| x.convert()).collect())
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

    network.send_next(message.to_network()).await?;

    // receiving from previous party
    let serialized_other_shares = network.receive_prev().await;
    let c = NetworkValue::from_network(serialized_other_shares)
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
        network.send_next(RingElement32(x.b).to_network()).await?;
        let serialized_reply = network.receive_prev().await;
        let missing_share = match NetworkValue::from_network(serialized_reply) {
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
        network
            .send_next(T::new_network_vec(message).to_network())
            .await?;

        // receiving from previous party
        let shares_c = {
            let serialized_other_share = network.receive_prev().await;
            let net_message = NetworkValue::from_network(serialized_other_share)?;
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
                    .map(|x| ((1, 1), DistanceShare {
                        code_dot: x[0].clone(),
                        mask_dot: x[1].clone(),
                    }))
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

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_additive(session: &mut Session, x: Vec<RingElement<u16>>) -> Result<Vec<u16>> {
        let prev_role = session.prev_identity()?;
        let network = &mut session.network_session;

        network
            .send_next(NetworkValue::VecRing16(x.clone()).to_network())
            .await?;

        let message_bytes = NetworkValue::VecRing16(x.clone()).to_network();
        trace!(target: "searcher::network", action = "send", party = ?prev_role, bytes = message_bytes.len(), rounds = 0);

        network.send_prev(message_bytes).await?;

        let serialized_reply_0 = network.receive_prev().await;
        let serialized_reply_1 = network.receive_next().await;

        let missing_share_0 = match NetworkValue::from_network(serialized_reply_0) {
            Ok(NetworkValue::VecRing16(element)) => element,
            _ => bail!("Could not deserialize VecRingElement16"),
        };
        let missing_share_1 = match NetworkValue::from_network(serialized_reply_1) {
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
            let mut own_shares = vec![(first_entry[index].clone(), second_entry[index].clone())];
            own_shares.iter_mut().for_each(|(_x, y)| {
                y.code.preprocess_iris_code_query_share();
                y.mask.preprocess_mask_code_query_share();
            });
            let session = session.clone();
            jobs.spawn(async move {
                let mut player_session = session.lock().await;
                let own_shares = own_shares.iter().map(|(x, y)| (x, y)).collect_vec();
                let x = galois_ring_pairwise_distance(&mut player_session, &own_shares).await;
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

    #[test]
    fn test_convert_distances_to_query_grouped_usize() {
        // Create test data with multiple queries and rotations using usize
        let distances = vec![
            // Query 0, Rotation 0: distances [10, 20, 30]
            ((0, 0), vec![10usize, 20, 30]),
            // Query 0, Rotation 2: distances [40, 50] (intentionally skipping rotation 1 to test ordering)
            ((0, 2), vec![40usize, 50]),
            // Query 0, Rotation 1: distances [60] (will be ordered correctly despite being added later)
            ((0, 1), vec![60usize]),
            // Query 1, Rotation 0: distances [100, 200, 300, 400]
            ((1, 0), vec![100usize, 200, 300, 400]),
            // Query 2, Rotation 1: distances [500, 600]
            ((2, 1), vec![500usize, 600]),
            // Query 2, Rotation 0: distances [700] (will be ordered before rotation 1)
            ((2, 0), vec![700usize]),
        ];

        let result = convert_distances_to_query_grouped(&distances);

        // Expected structure:
        // Query 0: [
        //   Rotation 0: [10, 20, 30],
        //   Rotation 1: [60],
        //   Rotation 2: [40, 50]
        // ]
        // Query 1: [
        //   Rotation 0: [100, 200, 300, 400]
        // ]
        // Query 2: [
        //   Rotation 0: [700],
        //   Rotation 1: [500, 600]
        // ]

        assert_eq!(result.len(), 3); // 3 queries

        // Query 0 checks
        assert_eq!(result[0].len(), 3); // 3 rotations
        assert_eq!(result[0][0], vec![10, 20, 30]); // rotation 0
        assert_eq!(result[0][1], vec![60]); // rotation 1
        assert_eq!(result[0][2], vec![40, 50]); // rotation 2

        // Query 1 checks
        assert_eq!(result[1].len(), 1); // 1 rotation
        assert_eq!(result[1][0], vec![100, 200, 300, 400]); // rotation 0

        // Query 2 checks
        assert_eq!(result[2].len(), 2); // 2 rotations
        assert_eq!(result[2][0], vec![700]); // rotation 0
        assert_eq!(result[2][1], vec![500, 600]); // rotation 1
    }
}
