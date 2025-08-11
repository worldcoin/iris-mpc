#[cfg(feature = "gpu_dependent")]
mod bucket_2d_threshold_test {
    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
    use eyre::Result;
    use iris_mpc_common::iris_db::iris::IrisCodeArray;
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, htod_on_stream_sync},
        threshold_ring::protocol::{ChunkShare, Circuits},
    };
    use itertools::{izip, Itertools};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use static_assertions::const_assert;
    use std::{env, sync::Arc};
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

    fn sample_code_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u32> {
        (0..size)
            .map(|_| {
                let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                let neg = rng.gen::<bool>();
                if neg {
                    x = (u16::MAX - x).wrapping_add(1);
                }
                x as u32
            })
            .collect::<Vec<_>>()
    }

    fn sample_mask_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u32> {
        (0..size)
            .map(|_| rng.gen_range::<u32, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u32))
            .collect::<Vec<_>>()
    }

    fn rep_share<R: Rng>(value: u32, rng: &mut R) -> (u32, u32, u32) {
        let a = rng.gen();
        let b = rng.gen();
        let c = value.wrapping_sub(a).wrapping_sub(b);

        (a, b, c)
    }

    fn rep_share_vec<R: Rng>(value: &[u32], rng: &mut R) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut a = Vec::with_capacity(value.len());
        let mut b = Vec::with_capacity(value.len());
        let mut c = Vec::with_capacity(value.len());
        for v in value.iter() {
            let (a_, b_, c_) = rep_share(*v, rng);
            a.push(a_);
            b.push(b_);
            c.push(c_);
        }
        (a, b, c)
    }

    fn to_gpu(
        a: &[u32],
        b: &[u32],
        devices: &[Arc<CudaDevice>],
        streams: &[CudaStream],
    ) -> Vec<ChunkShare<u32>> {
        debug_assert_eq!(a.len(), b.len());

        let mut result = Vec::with_capacity(devices.len());

        for (dev, stream, a, b) in izip!(
            devices,
            streams,
            a.chunks(INPUTS_PER_GPU_SIZE),
            b.chunks(INPUTS_PER_GPU_SIZE)
        ) {
            let a_ = htod_on_stream_sync(a, dev, stream).unwrap();
            let b_ = htod_on_stream_sync(b, dev, stream).unwrap();
            result.push(ChunkShare::new(a_, b_));
        }

        result
    }
    fn real_result(
        code_l: Vec<u32>,
        mask_l: Vec<u32>,
        code_r: Vec<u32>,
        mask_r: Vec<u32>,
    ) -> Vec<u32> {
        assert_eq!(code_l.len(), mask_l.len());
        assert_eq!(code_l.len(), code_r.len());
        assert_eq!(code_l.len(), mask_r.len());
        let mod_ = 1u64 << (16 + B_BITS);
        let mut result = Vec::with_capacity(THRESHOLDS.len() * THRESHOLDS.len());

        for t_l in THRESHOLDS {
            for t_r in THRESHOLDS {
                let a_l = Circuits::translate_threshold_a(t_l);
                let a_r = Circuits::translate_threshold_a(t_r);

                let mut count = 0;
                for (&c_l, &m_l, &c_r, &m_r) in
                    izip!(code_l.iter(), mask_l.iter(), code_r.iter(), mask_r.iter())
                {
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
        code_share_l_a: Vec<u32>,
        code_share_l_b: Vec<u32>,
        mask_share_l_a: Vec<u32>,
        mask_share_l_b: Vec<u32>,
        code_share_r_a: Vec<u32>,
        code_share_r_b: Vec<u32>,
        mask_share_r_a: Vec<u32>,
        mask_share_r_b: Vec<u32>,
        real_result: Vec<u32>,
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
        let code_l_gpu = to_gpu(&code_share_l_a, &code_share_l_b, &devices, &streams);
        let mask_l_gpu = to_gpu(&mask_share_l_a, &mask_share_l_b, &devices, &streams);
        let code_r_gpu = to_gpu(&code_share_r_a, &code_share_r_b, &devices, &streams);
        let mask_r_gpu = to_gpu(&mask_share_r_a, &mask_share_r_b, &devices, &streams);
        tracing::info!("id: {}, Data is on GPUs!", id);
        tracing::info!("id: {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            let code_l_gpu = code_l_gpu.iter().map(|x| x.as_view()).collect_vec();
            let mask_l_gpu = mask_l_gpu.iter().map(|x| x.as_view()).collect_vec();
            let code_r_gpu = code_r_gpu.iter().map(|x| x.as_view()).collect_vec();
            let mask_r_gpu = mask_r_gpu.iter().map(|x| x.as_view()).collect_vec();
            party.synchronize_streams(&streams);

            let now = Instant::now();
            let result = party.compare_multiple_thresholds_2d(
                &code_l_gpu,
                &mask_l_gpu,
                &code_r_gpu,
                &mask_r_gpu,
                &streams,
                &threshold,
            );

            party.synchronize_streams(&streams);
            tracing::info!("id: {}, Starting tests...", id);
            tracing::info!("id: {}, compute time: {:?}", id, now.elapsed());

            let mut correct = true;
            for (i, (r, r_)) in izip!(&result, &real_result).enumerate() {
                if r != r_ {
                    correct = false;
                    tracing::error!("id: {}, Test failed on index: {}: {} != {}", id, i, r, r_);
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
    async fn test_bucket_threshold() -> Result<()> {
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
        let code_dots_l = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let mask_dots_l = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let code_dots_r = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let mask_dots_r = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (code_share_l_a, code_share_l_b, code_share_l_c) =
            rep_share_vec(&code_dots_l, &mut rng);
        let (mask_share_l_a, mask_share_l_b, mask_share_l_c) =
            rep_share_vec(&mask_dots_l, &mut rng);
        let (code_share_r_a, code_share_r_b, code_share_r_c) =
            rep_share_vec(&code_dots_r, &mut rng);
        let (mask_share_r_a, mask_share_r_b, mask_share_r_c) =
            rep_share_vec(&mask_dots_r, &mut rng);
        let real_result = real_result(code_dots_l, mask_dots_l, code_dots_r, mask_dots_r);
        tracing::info!("Random shared inputs generated!");

        let code_share_l_a_ = code_share_l_a.to_owned();
        let code_share_l_b_ = code_share_l_b.to_owned();
        let code_share_l_c_ = code_share_l_c.to_owned();
        let mask_share_l_a_ = mask_share_l_a.to_owned();
        let mask_share_l_b_ = mask_share_l_b.to_owned();
        let mask_share_l_c_ = mask_share_l_c.to_owned();
        let code_share_r_a_ = code_share_r_a.to_owned();
        let code_share_r_b_ = code_share_r_b.to_owned();
        let code_share_r_c_ = code_share_r_c.to_owned();
        let mask_share_r_a_ = mask_share_r_a.to_owned();
        let mask_share_r_b_ = mask_share_r_b.to_owned();
        let mask_share_r_c_ = mask_share_r_c.to_owned();
        let real_result_ = real_result.to_owned();
        let real_result__ = real_result.to_owned();

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

            testcase(
                party,
                code_share_l_a,
                code_share_l_c,
                mask_share_l_a,
                mask_share_l_c,
                code_share_r_a,
                code_share_r_c,
                mask_share_r_a,
                mask_share_r_c,
                real_result,
            );
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

            testcase(
                party,
                code_share_l_b,
                code_share_l_a_,
                mask_share_l_b,
                mask_share_l_a_,
                code_share_r_b,
                code_share_r_a_,
                mask_share_r_b,
                mask_share_r_a_,
                real_result_,
            );
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

            testcase(
                party,
                code_share_l_c_,
                code_share_l_b_,
                mask_share_l_c_,
                mask_share_l_b_,
                code_share_r_c_,
                code_share_r_b_,
                mask_share_r_c_,
                mask_share_r_b_,
                real_result__,
            );
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
