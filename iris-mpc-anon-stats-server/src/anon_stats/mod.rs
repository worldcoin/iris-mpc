use eyre::Result;
use iris_mpc_common::{
    helpers::statistics::BucketStatistics, iris_db::iris::MATCH_THRESHOLD_RATIO,
};
use iris_mpc_cpu::{
    execution::session::Session,
    protocol::{
        anon_stats::compare_min_threshold_buckets,
        ops::{batch_signed_lift_vec, open_ring, translate_threshold_a},
    },
    shares::share::DistanceShare,
};
use itertools::Itertools;

use crate::anon_stats::types::{AnonStats1DMapping, AnonStatsOrigin};

pub mod store;
pub mod types;

pub fn calculate_threshold_a(n_buckets: usize) -> Vec<u32> {
    (1..=n_buckets)
        .map(|x: usize| {
            translate_threshold_a(MATCH_THRESHOLD_RATIO / (n_buckets as f64) * (x as f64))
        })
        .collect_vec()
}

pub async fn lift_bundles_1d(
    session: &mut Session,
    bundles: &[types::DistanceBundle1D],
) -> Result<Vec<types::LiftedDistanceBundle1D>> {
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

pub async fn process_1d_anon_stats_job(
    session: &mut Session,
    job: AnonStats1DMapping,
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
    let mut anon_stats =
        BucketStatistics::new(job_size, config.n_buckets_1d, config.party_id, origin.side);
    anon_stats.fill_buckets(&buckets, MATCH_THRESHOLD_RATIO, None);
    Ok(anon_stats)
}

pub mod test_helper {

    use iris_mpc_cpu::shares::{share::DistanceShare, RingElement, Share};
    use itertools::Itertools;

    use crate::anon_stats::types::DistanceBundle1D;

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
                    for (i, &threshold) in translated_thresholds.iter().enumerate() {
                        let dist = 0.5f64 - (code_dist as f64) / (2f64 * mask_dist as f64);
                        // let diff = (mask_dist * threshold).wrapping_sub(code_dist * 2u32.pow(16));
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
