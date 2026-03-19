use std::ops::Not;

use ampc_actor_utils::{
    execution::session::Session,
    protocol::{
        binary::{extract_msb_batch, lift, mul_lift_2k_to_32, open_bin, single_extract_msb},
        ops::{cross_mul, oblivious_cross_compare},
    },
};
use ampc_secret_sharing::{
    shares::{bit::Bit, DistanceShare, VecShare},
    Share,
};
use eyre::{eyre, Result};

use crate::protocol::{
    min_round_robin::{min_round_robin_batch_with, CrossCompareFnResult},
    ops::{DistancePair, B, B_BITS},
};

use iris_mpc_common::iris_db::iris::Threshold;

pub(crate) const MATCH_THRESHOLD_RATIO: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;

/// Precomputed threshold constant: A = (1 - 2*ratio) * B.
fn threshold_a(threshold: Threshold) -> u64 {
    ((1. - 2. * threshold.ratio()) * B as f64) as u64
}

/// Compares the distance between two iris pairs to a threshold.
///
/// - Takes as input two code and mask dot products between two irises,
///   i.e., code_dist = <iris1.code, iris2.code> and mask_dist = <iris1.mask, iris2.mask>,
///   already lifted to 32 bits if they are originally 16-bit.
/// - Multiplies with predefined threshold constants B = 2^16 and A = ((1. - 2.
///   * threshold_ratio) * B as f64).
/// - Compares mask_dist * A > code_dist * B.
/// - This corresponds to "distance > threshold", that is NOT match.
pub async fn greater_than(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
    threshold: Threshold,
) -> Result<Vec<Share<Bit>>> {
    let a = threshold_a(threshold) as u32;
    let diffs: Vec<Share<u32>> = distances
        .iter()
        .map(|d| {
            let x = d.mask_dot * a;
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
    let mut y = mul_lift_2k_to_32::<B_BITS>(&code_dist);
    let mut x = lift(session, VecShare::new_vec(vec![mask_dist])).await?;
    let mut x = x
        .pop()
        .ok_or(eyre!("Expected a single element in the VecShare"))?;
    x *= A as u32;
    y -= x;

    single_extract_msb(session, y).await
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
    distances: &[DistancePair<u32>],
) -> Result<Vec<bool>> {
    // d2.code_dot * d1.mask_dot - d1.code_dot * d2.mask_dot
    let diff = cross_mul(session, distances).await?;
    // Compute the MSB of the above
    let bits = extract_msb_batch(session, &diff).await?;
    // Open the MSB
    let opened_b = open_bin(session, &bits).await?;
    opened_b.into_iter().map(|x| Ok(x.convert())).collect()
}

// Box the future returned by the comparison function to make it easier to pass into min_round_robin_batch_with.
fn fhd_oblivious_cross_compare_boxed<'a>(
    session: &'a mut Session,
    pairs: &'a [DistancePair<u32>],
) -> CrossCompareFnResult<'a> {
    Box::pin(oblivious_cross_compare(session, pairs))
}

// Round-robin minimum distance selection for FHD.
pub(crate) async fn min_round_robin_batch(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
    batch_size: usize,
) -> Result<Vec<DistanceShare<u32>>> {
    min_round_robin_batch_with(
        session,
        distances,
        batch_size,
        fhd_oblivious_cross_compare_boxed,
    )
    .await
}

/// Compares distances to the given threshold and reveals "less than or equal".
pub async fn lte_and_open(
    session: &mut Session,
    distances: &[DistanceShare<u32>],
    threshold: Threshold,
) -> Result<Vec<bool>> {
    let bits = greater_than(session, distances, threshold).await?;
    open_bin(session, &bits)
        .await
        .map(|v| v.into_iter().map(|x| x.convert().not()).collect())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use super::*;
    use aes_prng::AesRng;
    use ampc_actor_utils::{
        execution::{
            local::{generate_local_identities, LocalRuntime},
            session::SessionHandles,
        },
        network::value::NetworkValue::RingElement32,
        protocol::{ops::batch_signed_lift_vec, test_utils::create_array_sharing},
    };
    use ampc_secret_sharing::RingElement;
    use eyre::bail;
    use rand::SeedableRng;
    use tokio::{sync::Mutex, task::JoinSet};
    use tracing::instrument;

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_single(session: &mut Session, x: Share<u32>) -> Result<RingElement<u32>> {
        let network = &mut session.network_session;
        network.send_next(RingElement32(x.b)).await?;
        let missing_share = match network.receive_prev().await {
            Ok(RingElement32(element)) => element,
            _ => bail!("Could not deserialize RingElement32"),
        };
        let (a, b) = x.get_ab();
        Ok(a + b + missing_share)
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
}
