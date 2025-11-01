use crate::{
    execution::session::Session,
    protocol::{
        binary::{bit_inject_ot_2round, extract_msb_u32_batch},
        ops::{min_of_pair_batch, B},
    },
    shares::{share::DistanceShare, vecshare::VecShare, Share},
};
use eyre::{eyre, Result};
use itertools::Itertools;

/// Compares the distance between two iris pairs to a list of thresholds, represented as t_i/B, with B = 2^16.
/// Use the [translate_threshold_a](crate::protocol::ops::translate_threshold_a) function to compute the A term of the threshold comparison.
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
/// Use the [translate_threshold_a](crate::protocol::ops::translate_threshold_a) function to compute the A term of the threshold comparison.
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use super::{compare_min_threshold_buckets, compare_threshold_buckets};
    use aes_prng::AesRng;
    use itertools::Itertools;
    use rand::{Rng, RngCore, SeedableRng};
    use rand_distr::{Distribution, Standard};
    use tokio::{sync::Mutex, task::JoinSet};

    use crate::{
        execution::{
            local::{generate_local_identities, LocalRuntime},
            session::{Session, SessionHandles},
        },
        protocol::ops::{open_ring, translate_threshold_a},
        shares::{share::DistanceShare, IntRing2k, RingElement, Share},
    };

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
    async fn test_compare_threshold_buckets() {
        const NUM_BUCKETS: usize = 100;
        const NUM_ITEMS: usize = 20;
        let mut rng = AesRng::seed_from_u64(0_u64);
        let items = (0..NUM_ITEMS)
            .flat_map(|_| {
                let mask = rng.gen_range(6000u32..12000);
                let code = rng.gen_range(-12000i16..12000);
                [code as u32, mask]
            })
            .collect_vec();

        let shares = create_array_sharing(&mut rng, &items);

        let thresholds: [f64; NUM_BUCKETS] =
            std::array::from_fn(|i| i as f64 / (NUM_BUCKETS * 2) as f64);
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
                    let diff = (mask_dist * threshold_a)
                        .wrapping_sub(code_dist.wrapping_mul(2u32.pow(16)));
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
                [code as u32, mask]
            })
            .collect_vec();

        let shares = create_array_sharing(&mut rng, &items);

        let thresholds: [f64; NUM_BUCKETS] =
            std::array::from_fn(|i| i as f64 / (NUM_BUCKETS * 2) as f64);
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
                        if (1f64 - a.0 as i32 as f64 / a.1 as f64)
                            < (1f64 - b.0 as i32 as f64 / b.1 as f64)
                        {
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
                    let diff = (mask_dist * threshold_a)
                        .wrapping_sub(code_dist.wrapping_mul(2u32.pow(16)));
                    acc[i] += if (diff as i32) < 0 { 1 } else { 0 };
                }
                acc
            });
        assert_eq!(t1, expected);
        assert_eq!(t2, expected);
        assert_eq!(t3, expected);
    }
}
