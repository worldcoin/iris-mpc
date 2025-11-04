use eyre::Result;
use iris_mpc_common::{
    anon_stats::{AnonStatsContext, AnonStatsMapping, AnonStatsOrigin},
    helpers::statistics::{AnonStatsResultSource, BucketStatistics, BucketStatistics2D},
    iris_db::iris::MATCH_THRESHOLD_RATIO,
};
use iris_mpc_cpu::{
    execution::session::Session,
    protocol::{
        anon_stats::{
            compare_against_thresholds_batched, compare_min_threshold_buckets,
            reduce_to_min_distance_batch,
        },
        ops::{
            batch_signed_lift_vec, open_ring, open_ring_element_broadcast, translate_threshold_a,
        },
    },
    shares::{share::DistanceShare, RingElement},
};
use itertools::{izip, Itertools};

pub type DistanceBundle1D = Vec<DistanceShare<u16>>;
pub type LiftedDistanceBundle1D = Vec<DistanceShare<u32>>;
pub type DistanceBundle2D = (DistanceBundle1D, DistanceBundle1D);
pub type LiftedDistanceBundle2D = (LiftedDistanceBundle1D, LiftedDistanceBundle1D);

pub fn calculate_threshold_a(n_buckets: usize) -> Vec<u32> {
    (1..=n_buckets)
        .map(|x: usize| {
            translate_threshold_a(MATCH_THRESHOLD_RATIO / (n_buckets as f64) * (x as f64))
        })
        .collect_vec()
}

pub async fn lift_bundles_1d(
    session: &mut Session,
    bundles: &[DistanceBundle1D],
) -> Result<Vec<LiftedDistanceBundle1D>> {
    let flattened = bundles
        .iter()
        .flat_map(|x| {
            x.iter()
                .flat_map(|y| [y.code_dot.clone(), y.mask_dot.clone()])
        })
        .collect_vec();
    let lifted_flattened = batch_signed_lift_vec(session, flattened.clone()).await?;

    // reconstruct lifted bundles in original shape
    let mut idx = 0;
    let lifted = bundles
        .iter()
        .map(|b| b.len())
        .map(|chunk_size| {
            let mut lifted_bundle = Vec::with_capacity(chunk_size);
            for _ in 0..chunk_size {
                let code_dot = lifted_flattened[idx].clone();
                let mask_dot = lifted_flattened[idx + 1].clone();
                idx += 2;
                lifted_bundle.push(DistanceShare { code_dot, mask_dot });
            }
            lifted_bundle
        })
        .collect();
    assert!(idx == lifted_flattened.len());
    Ok(lifted)
}

pub async fn lift_bundles_2d(
    session: &mut Session,
    bundles: &[DistanceBundle2D],
) -> Result<Vec<LiftedDistanceBundle2D>> {
    let bundle_left = bundles.iter().map(|(left, _)| left.clone()).collect_vec();
    let bundle_right = bundles.iter().map(|(_, right)| right.clone()).collect_vec();
    let lifted_left = lift_bundles_1d(session, &bundle_left).await?;
    let lifted_right = lift_bundles_1d(session, &bundle_right).await?;
    let lifted_bundles = lifted_left
        .into_iter()
        .zip(lifted_right.into_iter())
        .collect_vec();
    Ok(lifted_bundles)
}

pub async fn process_1d_anon_stats_job(
    session: &mut Session,
    job: AnonStatsMapping<DistanceBundle1D>,
    origin: &AnonStatsOrigin,
    config: &crate::config::AnonStatsServerConfig,
) -> Result<BucketStatistics> {
    let job_size = job.len();
    let job_data = job.into_bundles();
    let lifted_data = lift_bundles_1d(session, &job_data).await?;
    let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);

    // execute anon stats MPC protocol
    let bucket_result_shares = compare_min_threshold_buckets(
        session,
        translated_thresholds.as_slice(),
        lifted_data.as_slice(),
    )
    .await?;

    let buckets = open_ring(session, &bucket_result_shares).await?;
    let mut anon_stats = BucketStatistics::new(
        job_size,
        config.n_buckets_1d,
        config.party_id,
        origin.side.expect("1d stats need a side"),
    );
    anon_stats.fill_buckets(&buckets, MATCH_THRESHOLD_RATIO, None);
    anon_stats.source = AnonStatsResultSource::Aggregator;
    Ok(anon_stats)
}

pub async fn process_1d_lifted_anon_stats_job(
    session: &mut Session,
    job: AnonStatsMapping<LiftedDistanceBundle1D>,
    origin: &AnonStatsOrigin,
    config: &crate::config::AnonStatsServerConfig,
) -> Result<BucketStatistics> {
    let job_size = job.len();
    let job_data = job.into_bundles();
    let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);

    // execute anon stats MPC protocol
    let bucket_result_shares = compare_min_threshold_buckets(
        session,
        translated_thresholds.as_slice(),
        job_data.as_slice(),
    )
    .await?;

    let buckets = open_ring(session, &bucket_result_shares).await?;
    let mut anon_stats = BucketStatistics::new(
        job_size,
        config.n_buckets_1d,
        config.party_id,
        origin.side.expect("1d stats need a side"),
    );
    anon_stats.fill_buckets(&buckets, MATCH_THRESHOLD_RATIO, None);
    anon_stats.source = AnonStatsResultSource::Aggregator;
    Ok(anon_stats)
}

pub async fn process_2d_anon_stats_job(
    session: &mut Session,
    job: AnonStatsMapping<DistanceBundle2D>,
    config: &crate::config::AnonStatsServerConfig,
) -> Result<BucketStatistics2D> {
    let job_size = job.len();
    let job_data = job.into_bundles();
    let translated_thresholds = calculate_threshold_a(config.n_buckets_1d);

    let bundle_left = job_data.iter().map(|(left, _)| left.clone()).collect_vec();
    let bundle_right = job_data
        .iter()
        .map(|(_, right)| right.clone())
        .collect_vec();
    drop(job_data);

    // Lift both sides of the 2D bundles
    let lifted_left = lift_bundles_1d(session, &bundle_left).await?;
    drop(bundle_left);
    let lifted_right = lift_bundles_1d(session, &bundle_right).await?;
    drop(bundle_right);

    // Reduce both sides to min distances
    let lifted_min_left = reduce_to_min_distance_batch(session, &lifted_left).await?;
    drop(lifted_left);
    let lifted_min_right = reduce_to_min_distance_batch(session, &lifted_right).await?;
    drop(lifted_right);

    // execute anon stats MPC protocol
    let comparisons_left = compare_against_thresholds_batched(
        session,
        translated_thresholds.as_slice(),
        &lifted_min_left,
    )
    .await?;
    drop(lifted_min_left);
    let comparisons_right = compare_against_thresholds_batched(
        session,
        translated_thresholds.as_slice(),
        &lifted_min_right,
    )
    .await?;
    drop(lifted_min_right);

    // combine left and right comparisons by doing an outer product
    // at the same time also do the summing

    let mut bucket_shares =
        vec![RingElement::<u32>::default(); config.n_buckets_1d * config.n_buckets_1d];

    // prepare the correlated randomness for the bucket sums
    // we want additive shares of 0
    for bucket in &mut bucket_shares {
        *bucket += session.prf.gen_zero_share();
    }

    let mut bucket_ids = 0;
    // TODO: This could be parallelized if needed
    for left_chunk in comparisons_left.chunks(job_size) {
        for right_chunk in comparisons_right.chunks(job_size) {
            assert!(left_chunk.len() == right_chunk.len());

            let product_sum = izip!(left_chunk, right_chunk)
                .map(|(left_share, right_share)| left_share * right_share)
                .fold(RingElement(0u32), |acc, x| acc + x);
            bucket_shares[bucket_ids] += product_sum;
            bucket_ids += 1;
        }
    }

    let buckets = open_ring_element_broadcast(session, &bucket_shares).await?;
    let mut anon_stats = BucketStatistics2D::new(job_size, config.n_buckets_1d, config.party_id);
    anon_stats.fill_buckets(&buckets, MATCH_THRESHOLD_RATIO, None);
    anon_stats.source = AnonStatsResultSource::Aggregator;
    Ok(anon_stats)
}

pub mod test_helper_1d {
    use iris_mpc_cpu::shares::{share::DistanceShare, RingElement, Share};
    use itertools::Itertools;

    use crate::anon_stats::DistanceBundle1D;

    pub struct TestDistances {
        pub distances: Vec<Vec<[i16; 2]>>,
        pub shares0: Vec<DistanceBundle1D>,
        pub shares1: Vec<DistanceBundle1D>,
        pub shares2: Vec<DistanceBundle1D>,
    }

    impl TestDistances {
        pub fn generate_ground_truth_input(
            rng: &mut impl rand::Rng,
            num_bundles: usize,
            max_rotaions: usize,
        ) -> TestDistances {
            let sizes = (0..num_bundles)
                .map(|_| rng.gen_range(1..=max_rotaions))
                .collect_vec();
            let flat_size = sizes.iter().sum::<usize>();

            let items = (0..flat_size)
                .map(|_| {
                    let mask = rng.gen_range(6000i16..12000);
                    let code = rng.gen_range(-12000i16..12000);
                    [code, mask]
                })
                .collect_vec();
            let (shares1, shares2, shares3): (Vec<_>, Vec<_>, Vec<_>) = items
                .iter()
                .map(|&x| {
                    let share1: u16 = rng.gen();
                    let share2: u16 = rng.gen();
                    let share3: u16 = (x[0] as u16).wrapping_sub(share1).wrapping_sub(share2);
                    let mshare1: u16 = rng.gen();
                    let mshare2: u16 = rng.gen();
                    let mshare3: u16 = (x[1] as u16).wrapping_sub(mshare1).wrapping_sub(mshare2);
                    (
                        DistanceShare {
                            code_dot: Share {
                                a: RingElement(share1),
                                b: RingElement(share3),
                            },
                            mask_dot: Share {
                                a: RingElement(mshare1),
                                b: RingElement(mshare3),
                            },
                        },
                        DistanceShare {
                            code_dot: Share {
                                a: RingElement(share2),
                                b: RingElement(share1),
                            },
                            mask_dot: Share {
                                a: RingElement(mshare2),
                                b: RingElement(mshare1),
                            },
                        },
                        DistanceShare {
                            code_dot: Share {
                                a: RingElement(share3),
                                b: RingElement(share2),
                            },
                            mask_dot: Share {
                                a: RingElement(mshare3),
                                b: RingElement(mshare2),
                            },
                        },
                    )
                })
                .multiunzip();

            let mut idx = 0;
            let bundles = sizes
                .iter()
                .map(|&size| {
                    let mut bundle = Vec::with_capacity(size);
                    for _ in 0..size {
                        bundle.push(items[idx]);
                        idx += 1;
                    }
                    bundle
                })
                .collect_vec();
            assert!(idx == items.len());

            let mut idx = 0;
            let bundle_shares1 = sizes
                .iter()
                .map(|&size| {
                    let mut bundle = Vec::with_capacity(size);
                    for _ in 0..size {
                        bundle.push(shares1[idx].clone());
                        idx += 1;
                    }
                    bundle
                })
                .collect_vec();
            assert!(idx == items.len());

            let mut idx = 0;
            let bundle_shares2 = sizes
                .iter()
                .map(|&size| {
                    let mut bundle = Vec::with_capacity(size);
                    for _ in 0..size {
                        bundle.push(shares2[idx].clone());
                        idx += 1;
                    }
                    bundle
                })
                .collect_vec();
            assert!(idx == items.len());

            let mut idx = 0;
            let bundle_shares3 = sizes
                .iter()
                .map(|&size| {
                    let mut bundle = Vec::with_capacity(size);
                    for _ in 0..size {
                        bundle.push(shares3[idx].clone());
                        idx += 1;
                    }
                    bundle
                })
                .collect_vec();
            assert!(idx == items.len());
            TestDistances {
                distances: bundles,
                shares0: bundle_shares1,
                shares1: bundle_shares2,
                shares2: bundle_shares3,
            }
        }

        pub fn ground_truth_buckets(&self, translated_thresholds: &[u32]) -> Vec<u32> {
            let num_buckets = translated_thresholds.len();
            let expected = self
                .distances
                .iter()
                .map(|group| {
                    // reduce distances in each group
                    group
                        .iter()
                        .reduce(|a, b| {
                            // plain distance formula is (0.5 - code/2*mask), the below is that multiplied by 2
                            if (1f64 - a[0] as f64 / a[1] as f64)
                                < (1f64 - b[0] as f64 / b[1] as f64)
                            {
                                a
                            } else {
                                b
                            }
                        })
                        .expect("Expected at least one distance in the group")
                })
                .fold(vec![0; num_buckets], |mut acc, x| {
                    let code_dist = x[0];
                    let mask_dist = x[1];
                    let dist = 0.5f64 - (code_dist as f64) / (2f64 * mask_dist as f64);
                    for (i, &threshold) in translated_thresholds.iter().enumerate() {
                        acc[i] += if dist < 0.5f64 - threshold as f64 / (2f64 * 65536f64) {
                            1
                        } else {
                            0
                        };
                    }
                    acc
                });
            expected
        }
    }
}

pub mod test_helper_2d {

    use iris_mpc_cpu::shares::{share::DistanceShare, RingElement, Share};
    use itertools::Itertools;

    use crate::anon_stats::DistanceBundle2D;

    pub struct TestDistances {
        #[allow(clippy::type_complexity)]
        pub distances: Vec<(Vec<[i16; 2]>, Vec<[i16; 2]>)>,
        pub shares0: Vec<DistanceBundle2D>,
        pub shares1: Vec<DistanceBundle2D>,
        pub shares2: Vec<DistanceBundle2D>,
    }

    impl TestDistances {
        pub fn generate_ground_truth_input(
            rng: &mut impl rand::Rng,
            num_bundles: usize,
            max_rotaions: usize,
        ) -> TestDistances {
            let sizes = (0..num_bundles)
                .map(|_| rng.gen_range(1..=max_rotaions))
                .collect_vec();
            let flat_size = sizes.iter().sum::<usize>();

            let mut generate_items = || {
                let items = (0..flat_size)
                    .map(|_| {
                        let mask = rng.gen_range(6000i16..12000);
                        let code = rng.gen_range(-12000i16..12000);
                        [code, mask]
                    })
                    .collect_vec();
                let (shares1, shares2, shares3): (Vec<_>, Vec<_>, Vec<_>) = items
                    .iter()
                    .map(|&x| {
                        let share1: u16 = rng.gen();
                        let share2: u16 = rng.gen();
                        let share3: u16 = (x[0] as u16).wrapping_sub(share1).wrapping_sub(share2);
                        let mshare1: u16 = rng.gen();
                        let mshare2: u16 = rng.gen();
                        let mshare3: u16 =
                            (x[1] as u16).wrapping_sub(mshare1).wrapping_sub(mshare2);
                        (
                            DistanceShare {
                                code_dot: Share {
                                    a: RingElement(share1),
                                    b: RingElement(share3),
                                },
                                mask_dot: Share {
                                    a: RingElement(mshare1),
                                    b: RingElement(mshare3),
                                },
                            },
                            DistanceShare {
                                code_dot: Share {
                                    a: RingElement(share2),
                                    b: RingElement(share1),
                                },
                                mask_dot: Share {
                                    a: RingElement(mshare2),
                                    b: RingElement(mshare1),
                                },
                            },
                            DistanceShare {
                                code_dot: Share {
                                    a: RingElement(share3),
                                    b: RingElement(share2),
                                },
                                mask_dot: Share {
                                    a: RingElement(mshare3),
                                    b: RingElement(mshare2),
                                },
                            },
                        )
                    })
                    .multiunzip();

                let mut idx = 0;
                let bundles = sizes
                    .iter()
                    .map(|&size| {
                        let mut bundle = Vec::with_capacity(size);
                        for _ in 0..size {
                            bundle.push(items[idx]);
                            idx += 1;
                        }
                        bundle
                    })
                    .collect_vec();
                assert!(idx == items.len());

                let mut idx = 0;
                let bundle_shares1 = sizes
                    .iter()
                    .map(|&size| {
                        let mut bundle = Vec::with_capacity(size);
                        for _ in 0..size {
                            bundle.push(shares1[idx].clone());
                            idx += 1;
                        }
                        bundle
                    })
                    .collect_vec();
                assert!(idx == items.len());

                let mut idx = 0;
                let bundle_shares2 = sizes
                    .iter()
                    .map(|&size| {
                        let mut bundle = Vec::with_capacity(size);
                        for _ in 0..size {
                            bundle.push(shares2[idx].clone());
                            idx += 1;
                        }
                        bundle
                    })
                    .collect_vec();
                assert!(idx == items.len());

                let mut idx = 0;
                let bundle_shares3 = sizes
                    .iter()
                    .map(|&size| {
                        let mut bundle = Vec::with_capacity(size);
                        for _ in 0..size {
                            bundle.push(shares3[idx].clone());
                            idx += 1;
                        }
                        bundle
                    })
                    .collect_vec();
                assert!(idx == items.len());
                (bundles, bundle_shares1, bundle_shares2, bundle_shares3)
            };
            let (items_left, shares0_left, shares1_left, shares2_left) = generate_items();
            let (items_right, shares0_right, shares1_right, shares2_right) = generate_items();

            let bundles = items_left.into_iter().zip(items_right).collect_vec();
            let bundle_shares0 = shares0_left.into_iter().zip(shares0_right).collect_vec();
            let bundle_shares1 = shares1_left.into_iter().zip(shares1_right).collect_vec();
            let bundle_shares2 = shares2_left.into_iter().zip(shares2_right).collect_vec();

            TestDistances {
                distances: bundles,
                shares0: bundle_shares0,
                shares1: bundle_shares1,
                shares2: bundle_shares2,
            }
        }

        pub fn ground_truth_buckets(&self, translated_thresholds: &[u32]) -> Vec<u32> {
            let num_buckets = translated_thresholds.len();
            let expected = self
                .distances
                .iter()
                .map(|(group_left, group_right)| {
                    // reduce distances in each group
                    let red_left = group_left
                        .iter()
                        .reduce(|a, b| {
                            // plain distance formula is (0.5 - code/2*mask), the below is that multiplied by 2
                            if (1f64 - a[0] as f64 / a[1] as f64)
                                < (1f64 - b[0] as f64 / b[1] as f64)
                            {
                                a
                            } else {
                                b
                            }
                        })
                        .expect("Expected at least one distance in the group");
                    let red_right = group_right
                        .iter()
                        .reduce(|a, b| {
                            // plain distance formula is (0.5 - code/2*mask), the below is that multiplied by 2
                            if (1f64 - a[0] as f64 / a[1] as f64)
                                < (1f64 - b[0] as f64 / b[1] as f64)
                            {
                                a
                            } else {
                                b
                            }
                        })
                        .expect("Expected at least one distance in the group");
                    (red_left, red_right)
                })
                .fold(vec![0; num_buckets * num_buckets], |mut acc, (x, y)| {
                    let code_dist_left = x[0];
                    let mask_dist_left = x[1];
                    let code_dist_right = y[0];
                    let mask_dist_right = y[1];
                    let dist_left =
                        0.5f64 - (code_dist_left as f64) / (2f64 * mask_dist_left as f64);
                    let dist_right =
                        0.5f64 - (code_dist_right as f64) / (2f64 * mask_dist_right as f64);
                    for (i, &threshold_left) in translated_thresholds.iter().enumerate() {
                        for (j, &threshold_right) in translated_thresholds.iter().enumerate() {
                            acc[i * num_buckets + j] += if (dist_left
                                < 0.5f64 - threshold_left as f64 / (2f64 * 65536f64))
                                && (dist_right
                                    < 0.5f64 - threshold_right as f64 / (2f64 * 65536f64))
                            {
                                1
                            } else {
                                0
                            };
                        }
                    }
                    acc
                });
            expected
        }
    }
}
