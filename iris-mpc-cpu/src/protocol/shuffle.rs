use eyre::Result;
use itertools::Itertools;
use rand::Rng;

use crate::{
    execution::session::{Session, SessionHandles},
    shares::{
        ring_impl::VecRingElement,
        share::{reconstruct_id_distance_vector, DistanceShare},
        RingElement, Share,
    },
    utils::constants::N_PARTIES,
};

/// Secret shared permutation used in the shuffle protocol
/// Each party holds two out of three shares of the permutation pi(x) = pi_12(pi_20(pi_01(x))),
/// i.e., party i holds pi_{i,i+1} and pi_{i-1,i} (indices modulo 3)
pub type Permutation = (Vec<u32>, Vec<u32>);

fn batch_shuffle(
    perm: &[u32],
    data: &[RingElement<u32>],
    batch_size: usize,
) -> Vec<RingElement<u32>> {
    let mut res = Vec::with_capacity(data.len());
    for i in perm {
        let start = (*i as usize) * batch_size;
        let end = start + batch_size;
        res.extend_from_slice(&data[start..end]);
    }
    res
}

fn shuffle_triplets(
    perm: Vec<u32>,
    data: VecRingElement<u32>,
    batch_size: usize,
) -> VecRingElement<u32> {
    let mut res = VecRingElement::with_capacity(data.len());
    for batch in data.0.chunks(batch_size * 3) {
        res.extend(batch_shuffle(&perm, batch, 3));
    }
    res
}

/// Perform a random shuffle of the input distances using the 3-party shuffle protocol
/// from https://dl.acm.org/doi/abs/10.1145/3460120.3484560.
///
/// `distances` contains secret shared distances as (secret shared id, DistanceShare<u32>) tuples.
///
/// Protocol description:
/// Setup: input distances are shared among 3 parties: party 0 holds (A, B), party 1 holds (B, C), party 2 holds (C, A).
/// 0. Each party generates shares of a random permutation: pi_{i,i+1} and pi_{i-1,i} and correlated randomness Z_{i,i+1} and Z_{i-1,i} shared with party i+1 and party i-1, respectively.
/// 1. Party 0:
/// - Computes X0 = pi_01(A + B + Z_01),
/// - Computes X1 = pi_20(X0 + Z_20) and sends it to party 1,
/// - Generates random shares tilde(A) and tilde(B) of the output distances, which are shared with party 2 and party 1, respectively.
/// 2. Party 1:
/// - Computes Y0 = pi_01(C - Z_01) and sends it to party 2,
/// - Receives X1 from party 0,
/// - Computes X2 = pi_12(X1 + Z_12),
/// - Computes tilde(C_1) = X2 - tilde(B), where tilde(B) is correlated randomness shared with party 0,
/// 3. Party 2:
/// - Receives Y0 from party 1,
/// - Computes Y1 = pi_20(Y0 - Z_20),
/// - Computes Y2 = pi_12(Y1 - Z_12),
/// - Computes tilde(C_2) = Y2 - tilde(A), where tilde(A) is correlated randomness shared with party 0,
/// 4. Party 1 and Party 2 exchange tilde(C_1) and tilde(C_2) to reconstruct tilde(C).
/// 5. Each party outputs its shares of the shuffled distances as
/// - (tilde(A), tilde(B)) for party 0,
/// - (tilde(B), tilde(C)) for party 1,
/// - (tilde(C), tilde(A)) for party 2.
///
/// The protocol is correct because:
/// tilde(A) + tilde(B) + tilde(C) =
/// = tilde(A) + tilde(B) + tilde(C_1) + tilde(C_2)
/// = tilde(A) + tilde(B) + X2 - tilde(B) + Y2 - tilde(A)
/// = X2 + Y2
/// = pi_12(X1 + Z_12) + pi_12(Y1 - Z_12)
/// = pi_12(X1 + Y1)
/// = pi_12(pi_20(X0 + Z_20) + pi_20(Y0 - Z_20))
/// = pi_12(pi_20(X0 + Y0))
/// = pi_12(pi_20(pi_01(A + B + Z_01) + pi_01(C - Z_01)))
/// = pi_12(pi_20(pi_01(A + B + C)))
/// = pi(A + B + C)
/// = pi(distances)
#[allow(dead_code)]
pub(crate) async fn random_shuffle_batch(
    session: &mut Session,
    distances: Vec<Vec<(Share<u32>, DistanceShare<u32>)>>,
) -> Result<Vec<Vec<(Share<u32>, DistanceShare<u32>)>>> {
    // check that distances is not empty
    if distances.is_empty() {
        eyre::bail!("Input distances cannot be empty");
    }

    // check that all batches have the same length
    let batch_size = distances[0].len();
    if !distances.iter().all(|batch| batch.len() == batch_size) {
        eyre::bail!("All batches must have the same length");
    }

    let flattened_distances = distances.into_iter().flatten().collect_vec();

    let shuffle_role =
        (session.session_id().0 + session.own_role().index() as u32) % N_PARTIES as u32;

    let flattened_results = match shuffle_role {
        0 => shuffle_party_0(session, flattened_distances, batch_size).await?,
        1 => shuffle_party_1(session, flattened_distances, batch_size).await?,
        2 => shuffle_party_2(session, flattened_distances, batch_size).await?,
        _ => eyre::bail!("Invalid shuffle role: {}", shuffle_role),
    };

    // Reshape flattened_results back into batches
    let results = flattened_results
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(results)
}

async fn shuffle_party_0(
    session: &mut Session,
    distances: Vec<(Share<u32>, DistanceShare<u32>)>,
    batch_size: usize,
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    let n = distances.len();
    let prf = &mut session.prf;
    // Generate shares of a random permutation
    let (pi_01, pi_20) = prf.gen_permutation((batch_size) as u32)?;
    let (a, b): (Vec<Vec<_>>, Vec<Vec<_>>) = distances
        .into_iter()
        .map(|(id_share, dist_share)| {
            let a = vec![id_share.a, dist_share.code_dot.a, dist_share.mask_dot.a];
            let b = vec![id_share.b, dist_share.code_dot.b, dist_share.mask_dot.b];
            (a, b)
        })
        .unzip();
    let a: VecRingElement<u32> = a.into_iter().flatten().collect();
    let b: VecRingElement<u32> = b.into_iter().flatten().collect();

    let (z01, z20): (VecRingElement<u32>, VecRingElement<u32>) = (0..a.len())
        .map(|_| prf.gen_rands::<RingElement<u32>>())
        .unzip();

    // X0 = pi_01(A + B + Z_01)
    let x0 = shuffle_triplets(pi_01, ((a + b)? + z01)?, batch_size);
    // X1 = pi_20(X0 + Z_20)
    let x1 = shuffle_triplets(pi_20, (x0 + z20)?, batch_size);
    let network = &mut session.network_session;
    network.send_ring_vec_next(&x1).await?;

    // Generate tilde(B) and tilde(A)
    let (tilde_b, tilde_a) = (0..(3 * n))
        .map(|_| prf.gen_rands::<RingElement<u32>>())
        .unzip();
    Ok(reconstruct_id_distance_vector(tilde_b, tilde_a))
}

async fn shuffle_party_1(
    session: &mut Session,
    distances: Vec<(Share<u32>, DistanceShare<u32>)>,
    batch_size: usize,
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    let prf = &mut session.prf;
    // Generate shares of a random permutation
    // pi_12 and pi_01
    let (pi_12, pi_01) = prf.gen_permutation((batch_size) as u32)?;
    // extract C
    let c: VecRingElement<u32> = distances
        .into_iter()
        .flat_map(|(id_share, dist_share)| {
            [id_share.a, dist_share.code_dot.a, dist_share.mask_dot.a]
        })
        .collect();

    let (z12, z01): (VecRingElement<u32>, VecRingElement<u32>) = (0..c.len())
        .map(|_| prf.gen_rands::<RingElement<u32>>())
        .unzip();

    // Y0 = pi_01(C - Z_01)
    let y0 = shuffle_triplets(pi_01, (c - z01)?, batch_size);

    // ROUND 1
    // Send Y0 to party 2
    let network = &mut session.network_session;
    network.send_ring_vec_next(&y0).await?;

    // Receive X1 from party 0
    let x1 = network.receive_ring_vec_prev().await?;
    // X2 = pi_12(X1 + Z_12)
    let x2 = shuffle_triplets(pi_12, (x1 + z12)?, batch_size);

    // tilde(B) shared with party 0
    let tilde_b: VecRingElement<u32> = (0..x2.len())
        .map(|_| prf.get_prev_prf().gen::<RingElement<u32>>())
        .collect();
    // tilde(C_1) = X2 - tilde(B)
    let tilde_c1 = (x2 - &tilde_b)?;

    // ROUND 2
    // Send tilde(C_1) to party 2
    network.send_ring_vec_next(&tilde_c1).await?;

    // Receive tilde(C_2) from party 2
    let tilde_c2 = network.receive_ring_vec_next().await?;

    // Compute tilde(C) = tilde(C_1) + tilde(C_2)
    let tilde_c = (tilde_c1 + tilde_c2)?;

    // Reconstruct shares using tilde_C and tilde_B
    Ok(reconstruct_id_distance_vector(tilde_c, tilde_b))
}

async fn shuffle_party_2(
    session: &mut Session,
    distances: Vec<(Share<u32>, DistanceShare<u32>)>,
    batch_size: usize,
) -> Result<Vec<(Share<u32>, DistanceShare<u32>)>> {
    let n = distances.len();
    let prf = &mut session.prf;
    // Generate shares of a random permutation
    let (pi_20, pi_12) = prf.gen_permutation((batch_size) as u32)?;

    let (z20, z12): (VecRingElement<u32>, VecRingElement<u32>) = (0..3 * n)
        .map(|_| prf.gen_rands::<RingElement<u32>>())
        .unzip();

    // ROUND 1
    // Receive Y0 from party 1
    let network = &mut session.network_session;

    let y0 = network.receive_ring_vec_prev().await?;

    // Y1 = pi_20(Y0 - Z_20)
    let y1 = shuffle_triplets(pi_20, (y0 - z20)?, batch_size);
    // Y2 = pi_12(Y1 - Z_12)
    let y2 = shuffle_triplets(pi_12, (y1 - z12)?, batch_size);
    // tilde(A) shared with party 0
    let tilde_a: VecRingElement<u32> = (0..y2.len())
        .map(|_| prf.get_my_prf().gen::<RingElement<u32>>())
        .collect();
    // tilde(C_2) = Y2 - tilde(A)
    let tilde_c2 = (y2 - &tilde_a)?;

    // ROUND 2
    // Send tilde(C_2) to party 1
    network.send_ring_vec_prev(&tilde_c2).await?;

    // Receive tilde(C_1) from party 1
    let tilde_c1 = network.receive_ring_vec_prev().await?;

    // Compute tilde(C) = tilde(C_1) + tilde(C_2)
    let tilde_c = (tilde_c1 + tilde_c2)?;

    // Reconstruct shares using tilde_A and tilde_C
    Ok(reconstruct_id_distance_vector(tilde_a, tilde_c))
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    use crate::{hawkers::aby3::test_utils::setup_local_store_aby3_players, network::NetworkType};

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_shuffle() -> Result<()> {
        let num_batches = 3;
        let batch_size = 6_u32;
        // Create a sorted plain list of distances:
        // (0,(0,1)), (1,(1,1)), ..., (5,(5,1)) 1st batch
        // (6,(6,1)), (7,(7,1)), ..., (11,(11,1)) 2nd batch
        // (12,(12,1)), (13,(13,1)), ..., (17,(17,1)) 3rd batch
        let plain_list = (0..num_batches)
            .map(|batch_i| {
                (0..batch_size)
                    .map(|i| (batch_size * batch_i + i, (batch_size * batch_i + i, 1)))
                    .collect_vec()
            })
            .collect_vec();

        // PRF seeds are fixed in test setup, so the shuffle is deterministic
        let mut local_stores = setup_local_store_aby3_players(NetworkType::Local).await?;
        let mut jobs = JoinSet::new();
        for store in local_stores.iter_mut() {
            let store = store.clone();
            let plain_list = plain_list.clone();
            jobs.spawn(async move {
                let mut store_lock = store.lock().await;
                let role = store_lock.session.own_role();
                let distances = plain_list
                    .iter()
                    .map(|batch| {
                        batch
                            .iter()
                            .map(|(id, (code_dist, mask_dist))| {
                                (
                                    Share::from_const(*id, role),
                                    DistanceShare::new(
                                        Share::from_const(*code_dist, role),
                                        Share::from_const(*mask_dist, role),
                                    ),
                                )
                            })
                            .collect_vec()
                    })
                    .collect_vec();
                random_shuffle_batch(&mut store_lock.session, distances).await
            });
        }
        let res = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        assert_eq!(res.len(), N_PARTIES);
        assert_eq!(res[0].len(), num_batches as usize);
        assert_eq!(res[1].len(), num_batches as usize);
        assert_eq!(res[2].len(), num_batches as usize);

        let perm = [2, 4, 3, 1, 0, 5];
        let expected = (0..num_batches)
            .map(|batch_i| {
                (0..batch_size)
                    .map(|i| {
                        (
                            batch_size * batch_i + perm[i as usize],
                            (batch_size * batch_i + perm[i as usize], 1),
                        )
                    })
                    .collect_vec()
            })
            .collect_vec();

        for (batch_i, expected_batch) in expected.into_iter().enumerate() {
            for (i, expected_i) in expected_batch.into_iter().enumerate() {
                let distance_i = {
                    let mut id = Share::zero();
                    let mut dist = DistanceShare::new(Share::zero(), Share::zero());
                    for party_res in res.iter() {
                        id += party_res[batch_i][i].clone().0;
                        dist += &party_res[batch_i][i].clone().1;
                    }
                    let id = id.get_a().convert();
                    let code_dot = dist.code_dot.get_a().convert();
                    let mask_dot = dist.mask_dot.get_a().convert();

                    (id, (code_dot, mask_dot))
                };
                assert_eq!(distance_i, expected_i);
            }
        }

        Ok(())
    }
}
