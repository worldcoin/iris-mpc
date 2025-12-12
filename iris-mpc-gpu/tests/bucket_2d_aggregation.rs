#[cfg(feature = "gpu_dependent")]
mod bucket_2d_aggregation_test {
    use ampc_anon_stats::AnonStatsOperation::Uniqueness;
    use cudarc::{driver::CudaStream, nccl::Id};
    use eyre::Result;
    use iris_mpc_common::iris_db::iris::IrisCodeArray;
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
        server::anon_stats::{CpuDistanceShare, TwoSidedDistanceCache, TwoSidedMinDistanceCache},
        threshold_ring::protocol::Circuits,
    };
    use itertools::{izip, Itertools};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use static_assertions::const_assert;
    use std::{collections::HashMap, env, sync::Arc};
    use tokio::time::Instant;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    const fn gen_thresholds<const N: usize>() -> [f64; N] {
        let mut thresholds = [0.0; N];
        let step = 0.375 / (N as f64);
        let mut i = 0;
        while i < N {
            thresholds[i] = step * (i + 1) as f64;
            i += 1;
        }
        thresholds
    }

    const DB_RNG_SEED: u64 = 0xdeadbeef;
    const INPUTS_PER_GPU_SIZE: usize = 64;
    const THRESHOLDS: [f64; 25] = gen_thresholds();

    const B_BITS: u64 = 16;
    pub(crate) const B: u64 = 1 << B_BITS;

    #[allow(clippy::type_complexity)]
    fn sample_cpu_distancs<R: Rng>(
        size: usize,
        rng: &mut R,
    ) -> HashMap<u64, (Vec<(u16, u16)>, Vec<(u16, u16)>)> {
        (0..size)
            .map(|idx| {
                let num = rng.gen_range(1usize..=4);
                let left = (0..num)
                    .map(|_| {
                        let mut code =
                            rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                        let neg = rng.gen::<bool>();
                        if neg {
                            code = (u16::MAX - code).wrapping_add(1);
                        }
                        let mask =
                            rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                        (code, mask)
                    })
                    .collect();
                let num = rng.gen_range(1usize..=4);
                let right = (0..num)
                    .map(|_| {
                        let mut code =
                            rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                        let neg = rng.gen::<bool>();
                        if neg {
                            code = (u16::MAX - code).wrapping_add(1);
                        }
                        let mask =
                            rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                        (code, mask)
                    })
                    .collect();
                (idx as u64, (left, right))
            })
            .collect()
    }

    fn rep_share<R: Rng>(value: u16, rng: &mut R) -> (u16, u16, u16) {
        let a = rng.gen();
        let b = rng.gen();
        let c = value.wrapping_sub(a).wrapping_sub(b);

        (a, b, c)
    }

    #[allow(clippy::type_complexity)]
    fn rep_share_distance_cache<R: Rng>(
        value: &HashMap<u64, (Vec<(u16, u16)>, Vec<(u16, u16)>)>,
        rng: &mut R,
    ) -> (
        TwoSidedDistanceCache,
        TwoSidedDistanceCache,
        TwoSidedDistanceCache,
    ) {
        let mut a = TwoSidedDistanceCache::default();
        let mut b = TwoSidedDistanceCache::default();
        let mut c = TwoSidedDistanceCache::default();
        for (key, (left, right)) in value.iter() {
            let (left_shared_a, left_shared_b, left_shared_c) = left
                .iter()
                .enumerate()
                .map(|(idx, (c, m))| {
                    let (ca, cb, cc) = rep_share(*c, rng);
                    let (ma, mb, mc) = rep_share(*m, rng);
                    (
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: ca,
                            code_b: cc,
                            mask_a: ma,
                            mask_b: mc,
                            operation: Uniqueness,
                        },
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: cb,
                            code_b: ca,
                            mask_a: mb,
                            mask_b: ma,
                            operation: Uniqueness,
                        },
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: cc,
                            code_b: cb,
                            mask_a: mc,
                            mask_b: mb,
                            operation: Uniqueness,
                        },
                    )
                })
                .collect();
            let (right_shared_a, right_shared_b, right_shared_c) = right
                .iter()
                .enumerate()
                .map(|(idx, (c, m))| {
                    let (ca, cb, cc) = rep_share(*c, rng);
                    let (ma, mb, mc) = rep_share(*m, rng);
                    (
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: ca,
                            code_b: cc,
                            mask_a: ma,
                            mask_b: mc,
                            operation: Uniqueness,
                        },
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: cb,
                            code_b: ca,
                            mask_a: mb,
                            mask_b: ma,
                            operation: Uniqueness,
                        },
                        CpuDistanceShare {
                            idx: idx as u64,
                            code_a: cc,
                            code_b: cb,
                            mask_a: mc,
                            mask_b: mb,
                            operation: Uniqueness,
                        },
                    )
                })
                .collect();

            a.map.insert(*key, (left_shared_a, right_shared_a));
            b.map.insert(*key, (left_shared_b, right_shared_b));
            c.map.insert(*key, (left_shared_c, right_shared_c));
        }
        (a, b, c)
    }

    #[allow(clippy::type_complexity)]
    fn real_result(
        distances: HashMap<u64, (Vec<(u16, u16)>, Vec<(u16, u16)>)>,
    ) -> Vec<(u32, u32, u32, u32)> {
        distances
            .into_iter()
            .sorted_by_key(|(key, _)| *key)
            .map(|(_, (left, right))| {
                let left_reduced = left
                    .into_iter()
                    .reduce(|(c1, m1), (c2, m2)| {
                        if (c1 as i16 as i64 * m2 as i64) > (c2 as i16 as i64 * m1 as i64) {
                            (c1, m1)
                        } else {
                            (c2, m2)
                        }
                    })
                    .unwrap();
                let right_reduced = right
                    .into_iter()
                    .reduce(|(c1, m1), (c2, m2)| {
                        if (c1 as i16 as i64 * m2 as i64) > (c2 as i16 as i64 * m1 as i64) {
                            (c1, m1)
                        } else {
                            (c2, m2)
                        }
                    })
                    .unwrap();
                (
                    left_reduced.0 as i16 as i32 as u32,
                    left_reduced.1 as i16 as i32 as u32,
                    right_reduced.0 as i16 as i32 as u32,
                    right_reduced.1 as i16 as i32 as u32,
                )
            })
            .collect()
    }
    fn real_result_2(inputs: &[(u32, u32, u32, u32)]) -> Vec<u32> {
        let mod_ = 1u64 << (16 + B_BITS);
        let mut result = Vec::with_capacity(THRESHOLDS.len() * THRESHOLDS.len());

        for t_l in THRESHOLDS {
            for t_r in THRESHOLDS {
                let a_l = Circuits::translate_threshold_a(t_l);
                let a_r = Circuits::translate_threshold_a(t_r);

                let mut count = 0;
                for &(c_l, m_l, c_r, m_r) in inputs.iter() {
                    let left = (((m_l as u64) * a_l)
                        .wrapping_sub((c_l as u64) << B_BITS)
                        .wrapping_sub(1))
                        % mod_;
                    let msb_l = (left >> (B_BITS + 16 - 1)) & 1 == 1;
                    let right = (((m_r as u64) * a_r)
                        .wrapping_sub((c_r as u64) << B_BITS)
                        .wrapping_sub(1))
                        % mod_;
                    let msb_r = (right >> (B_BITS + 16 - 1)) & 1 == 1;
                    count += (msb_l && msb_r) as u32;
                }
                result.push(count);
            }
        }
        result
    }

    fn open(
        party: &mut Circuits,
        min_distance_cache: &TwoSidedMinDistanceCache,
        streams: &[CudaStream],
    ) -> Vec<(u32, u32, u32, u32)> {
        let device = party.get_devices()[0].clone();
        let opened = [
            (
                &min_distance_cache.left_code_a,
                &min_distance_cache.left_code_b,
            ),
            (
                &min_distance_cache.left_mask_a,
                &min_distance_cache.left_mask_b,
            ),
            (
                &min_distance_cache.right_code_a,
                &min_distance_cache.right_code_b,
            ),
            (
                &min_distance_cache.right_mask_a,
                &min_distance_cache.right_mask_b,
            ),
        ]
        .into_iter()
        .map(|(a, b)| {
            let b_on_dev = htod_on_stream_sync(b, &device, &streams[0]).unwrap();
            let mut c_on_dev = htod_on_stream_sync(b, &device, &streams[0]).unwrap();
            cudarc::nccl::result::group_start().unwrap();
            party.comms()[0]
                .send(&b_on_dev, party.next_id(), &streams[0])
                .unwrap();
            party.comms()[0]
                .receive(&mut c_on_dev, party.prev_id(), &streams[0])
                .unwrap();
            cudarc::nccl::result::group_end().unwrap();
            let mut c = dtoh_on_stream_sync(&c_on_dev, &device, &streams[0]).unwrap();
            for (cc, aa, bb) in izip!(c.iter_mut(), a, b) {
                *cc = cc.wrapping_add(*aa).wrapping_add(*bb);
            }
            c
        })
        .collect::<Vec<_>>();

        izip!(
            opened[0].iter().copied(),
            opened[1].iter().copied(),
            opened[2].iter().copied(),
            opened[3].iter().copied()
        )
        .collect()
    }

    fn install_tracing() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    #[allow(clippy::too_many_arguments)]
    fn testcase(
        mut party: Circuits,
        distances: TwoSidedDistanceCache,
        real_result: Vec<(u32, u32, u32, u32)>,
        real_result2: Vec<u32>,
    ) {
        let id = party.peer_id();

        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        let mut threshold = Vec::with_capacity(THRESHOLDS.len());
        for t in THRESHOLDS {
            let a = ((1. - 2. * t) * (B as f64)) as u64;
            threshold.push(a as u16);
        }

        // Import to GPU
        tracing::info!("id: {}, Data is on GPUs!", id);
        tracing::info!("id: {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            party.synchronize_streams(&streams);
            let distances_run = vec![distances.clone()];

            let now = Instant::now();
            let min_distance_cache =
                TwoSidedDistanceCache::into_min_distance_cache(distances_run, &mut party, &streams);

            party.synchronize_streams(&streams);
            tracing::info!("id: {}, Starting tests...", id);
            tracing::info!("id: {}, compute time: {:?}", id, now.elapsed());

            tracing::info!("id: {}, opening", id);
            let result = open(&mut party, &min_distance_cache, &streams);
            tracing::info!("id: {}, opened", id);
            party.synchronize_streams(&streams);

            let buckets = min_distance_cache.compute_buckets(&mut party, &streams, &threshold);

            let mut correct = true;
            for (i, (r, r_)) in izip!(&result, &real_result).enumerate() {
                if r != r_ {
                    correct = false;
                    tracing::error!(
                        "id: {}, Test failed on index step1: {}: {:?} != {:?}",
                        id,
                        i,
                        r,
                        r_
                    );
                    error = true;
                    break;
                }
            }
            for (i, (r, r_)) in izip!(&buckets, &real_result2).enumerate() {
                if r != r_ {
                    correct = false;
                    tracing::error!(
                        "id: {}, Test failed on index step2: {}: {:?} != {:?}",
                        id,
                        i,
                        r,
                        r_
                    );
                    error = true;
                    break;
                }
            }
            if correct {
                tracing::info!("id: {}, Test passed!", id);
            }
        }
        assert!(!error);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn test_bucket_threshold_with_aggregation() -> Result<()> {
        install_tracing();
        env::set_var("NCCL_P2P_LEVEL", "LOC");
        env::set_var("NCCL_NET", "Socket");
        env::set_var("NCCL_P2P_DIRECT_DISABLE", "1");
        env::set_var("NCCL_SHM_DISABLE", "1");

        let chacha_seeds0 = ([0u32; 8], [2u32; 8]);
        let chacha_seeds1 = ([1u32; 8], [0u32; 8]);
        let chacha_seeds2 = ([2u32; 8], [1u32; 8]);

        const_assert!(
            INPUTS_PER_GPU_SIZE % (64) == 0,
            // Mod 16 for randomness
        );

        let mut rng = StdRng::seed_from_u64(DB_RNG_SEED);

        let device_manager = DeviceManager::init();
        let mut device_managers = device_manager
            .split_into_n_chunks(3)
            .expect("have at least 3 devices");
        let device_manager2 = Arc::new(device_managers.pop().unwrap());
        let device_manager1 = Arc::new(device_managers.pop().unwrap());
        let device_manager0 = Arc::new(device_managers.pop().unwrap());
        let n_devices = device_manager0.devices().len();
        let ids0 = (0..n_devices)
            .map(|_| Id::new().unwrap())
            .collect::<Vec<_>>();
        let ids1 = ids0.clone();
        let ids2 = ids0.clone();

        // Get inputs
        let distances = sample_cpu_distancs(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (distances_a, distances_b, distances_c) =
            rep_share_distance_cache(&distances, &mut rng);
        tracing::info!("Random shared inputs generated!");

        let real_result = real_result(distances);
        let real_result2 = real_result_2(&real_result);
        let real_result_ = real_result.to_owned();
        let real_result__ = real_result.to_owned();
        let real_result2_ = real_result2.to_owned();
        let real_result2__ = real_result2.to_owned();

        let task0 = tokio::task::spawn_blocking(move || {
            let comms0 = device_manager0
                .instantiate_network_from_ids(0, &ids0)
                .unwrap();

            let party = Circuits::new(
                0,
                INPUTS_PER_GPU_SIZE,
                INPUTS_PER_GPU_SIZE / 64,
                None,
                chacha_seeds0,
                device_manager0,
                comms0,
            );

            testcase(party, distances_a, real_result, real_result2);
        });

        let task1 = tokio::task::spawn_blocking(move || {
            let comms1 = device_manager1
                .instantiate_network_from_ids(1, &ids1)
                .unwrap();

            let party = Circuits::new(
                1,
                INPUTS_PER_GPU_SIZE,
                INPUTS_PER_GPU_SIZE / 64,
                None,
                chacha_seeds1,
                device_manager1,
                comms1,
            );

            testcase(party, distances_b, real_result_, real_result2_);
        });

        let task2 = tokio::task::spawn_blocking(move || {
            let comms2 = device_manager2
                .instantiate_network_from_ids(2, &ids2)
                .unwrap();

            let party = Circuits::new(
                2,
                INPUTS_PER_GPU_SIZE,
                INPUTS_PER_GPU_SIZE / 64,
                None,
                chacha_seeds2,
                device_manager2,
                comms2,
            );

            testcase(party, distances_c, real_result__, real_result2__);
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
