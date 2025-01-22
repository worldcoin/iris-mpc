#[cfg(feature = "gpu_dependent")]
mod buckets_test {
    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
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

    const DB_RNG_SEED: u64 = 0xdeadbeef;
    // ceil(930 * 125_000 / 2048) * 2048
    // const INPUTS_PER_GPU_SIZE: usize = 116_250_624;
    const INPUTS_PER_GPU_SIZE: usize = 12_507_136;

    const B_BITS: u64 = 16;
    pub(crate) const B: u64 = 1 << B_BITS;
    // pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as
    // u64;
    const THRESHOLDS: [f64; 4] = [0.1, 0.2, 0.3, 0.375];

    fn sample_code_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
        (0..size)
            .map(|_| {
                let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                let neg = rng.gen::<bool>();
                if neg {
                    x = u16::MAX - x + 1;
                }
                x
            })
            .collect::<Vec<_>>()
    }

    fn sample_mask_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
        (0..size)
            .map(|_| rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16))
            .collect::<Vec<_>>()
    }

    fn rep_share<R: Rng>(value: u16, rng: &mut R) -> (u16, u16, u16) {
        let a = rng.gen();
        let b = rng.gen();
        let c = value - a - b;

        (a, b, c)
    }

    fn rep_share_vec<R: Rng>(value: &[u16], rng: &mut R) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
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
        a: &[u16],
        b: &[u16],
        devices: &[Arc<CudaDevice>],
        streams: &[CudaStream],
    ) -> Vec<ChunkShare<u16>> {
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

    fn real_result_msb(code_input: Vec<u16>, mask_input: Vec<u16>) -> Vec<u32> {
        assert_eq!(code_input.len(), mask_input.len());
        let mod_ = 1u64 << (16 + B_BITS);
        let mut result = Vec::with_capacity(THRESHOLDS.len());

        for t in THRESHOLDS {
            let a = Circuits::translate_threshold_a(t);

            let mut count = 0;
            for (c, m) in code_input.iter().cloned().zip(mask_input.iter().cloned()) {
                let r = ((m as u64) * a - ((c as u64) << B_BITS) - 1) % mod_;
                let msb = r >> (B_BITS + 16 - 1) & 1 == 1;
                count += msb as u32;
            }
            result.push(count);
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

    fn testcase(
        mut party: Circuits,
        code_share_a: Vec<u16>,
        code_share_b: Vec<u16>,
        mask_share_a: Vec<u16>,
        mask_share_b: Vec<u16>,
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
        let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices, &streams);
        let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices, &streams);
        tracing::info!("id: {}, Data is on GPUs!", id);
        tracing::info!("id: {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            let a = devices[0].alloc_zeros::<u32>(THRESHOLDS.len()).unwrap();
            let b = devices[0].alloc_zeros::<u32>(THRESHOLDS.len()).unwrap();
            let mut bucket = ChunkShare::new(a, b);
            let code_gpu = code_gpu.iter().map(|x| x.as_view()).collect_vec();
            let mask_gpu = mask_gpu.iter().map(|x| x.as_view()).collect_vec();
            party.synchronize_streams(&streams);

            let now = Instant::now();
            party.compare_multiple_thresholds(
                &code_gpu,
                &mask_gpu,
                &streams,
                &threshold,
                &mut bucket,
            );

            party.synchronize_streams(&streams);
            tracing::info!("id: {}, Starting tests...", id);
            tracing::info!("id: {}, compute time: {:?}", id, now.elapsed());

            let now = Instant::now();
            let result = party.open_buckets(&bucket, &streams);
            party.synchronize_streams(&streams);
            tracing::info!("id: {}, Starting tests...", id);
            tracing::info!(
                "id: {}, Open and transfer to CPU time: {:?}",
                id,
                now.elapsed()
            );

            if result == real_result {
                tracing::info!("id: {}, Test passed!", id);
            } else {
                tracing::error!("id: {}, Test failed: {:?} != {:?}", id, real_result, result);
                error = true;
            }
        }
        assert!(!error);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn test_buckets() -> eyre::Result<()> {
        install_tracing();
        env::set_var("NCCL_P2P_LEVEL", "LOC");
        env::set_var("NCCL_NET", "Socket");
        env::set_var("NCCL_P2P_DIRECT_DISABLE", "1");
        env::set_var("NCCL_SHM_DISABLE", "1");

        let chacha_seeds0 = ([0u32; 8], [2u32; 8]);
        let chacha_seeds1 = ([1u32; 8], [0u32; 8]);
        let chacha_seeds2 = ([2u32; 8], [1u32; 8]);

        const_assert!(
            INPUTS_PER_GPU_SIZE % (2048) == 0,
            // Mod 16 for randomness, mod 64 for chunk size
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
        let code_dots = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let mask_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (code_share_a, code_share_b, code_share_c) = rep_share_vec(&code_dots, &mut rng);
        let (mask_share_a, mask_share_b, mask_share_c) = rep_share_vec(&mask_dots, &mut rng);
        let real_result = real_result_msb(code_dots, mask_dots);
        tracing::info!("Random shared inputs generated!");

        let code_share_a_ = code_share_a.to_owned();
        let code_share_b_ = code_share_b.to_owned();
        let code_share_c_ = code_share_c.to_owned();
        let mask_share_a_ = mask_share_a.to_owned();
        let mask_share_b_ = mask_share_b.to_owned();
        let mask_share_c_ = mask_share_c.to_owned();
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
                chacha_seeds0,
                device_manager0,
                comms0,
            );

            testcase(
                party,
                code_share_a,
                code_share_c,
                mask_share_a,
                mask_share_c,
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
                chacha_seeds1,
                device_manager1,
                comms1,
            );

            testcase(
                party,
                code_share_b,
                code_share_a_,
                mask_share_b,
                mask_share_a_,
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
                chacha_seeds2,
                device_manager2,
                comms2,
            );

            testcase(
                party,
                code_share_c_,
                code_share_b_,
                mask_share_c_,
                mask_share_b_,
                real_result__,
            );
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
