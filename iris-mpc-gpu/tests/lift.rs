#[cfg(feature = "gpu_dependent")]
mod lift_test {
    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
    use eyre::Result;
    use iris_mpc_common::iris_db::iris::IrisCodeArray;
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
        threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
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

    fn to_view<'a, T>(inp: &'a [ChunkShare<T>]) -> Vec<ChunkShareView<'a, T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.as_view());
        }
        res
    }

    fn sample_mask_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
        (0..size)
            .map(|_| rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16))
            .collect::<Vec<_>>()
    }

    fn rep_share<R: Rng>(value: u16, rng: &mut R) -> (u16, u16, u16) {
        let a = rng.gen();
        let b = rng.gen();
        let c = value.wrapping_sub(a).wrapping_sub(b);

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

    fn real_result_msb(mask_input: Vec<u16>) -> Vec<u32> {
        mask_input.into_iter().map(|x| (x as u32)).collect()
    }

    fn open(
        party: &mut Circuits,
        x: &mut [ChunkShareView<u32>],
        corrections: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) -> Vec<u32> {
        let n_devices = x.len();
        let mut res_a = Vec::with_capacity(n_devices);
        let mut res_b = Vec::with_capacity(n_devices);
        let mut res_c = Vec::with_capacity(n_devices);
        let mut corr_a = Vec::with_capacity(n_devices);
        let mut corr_b = Vec::with_capacity(n_devices);
        let mut corr_c = Vec::with_capacity(n_devices);

        let devices = party.get_devices();
        for (idx, (res, corr)) in izip!(x.iter(), corrections.iter()).enumerate() {
            res_a.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
            res_b.push(dtoh_on_stream_sync(&res.b, &devices[idx], &streams[idx]).unwrap());
            corr_a.push(dtoh_on_stream_sync(&corr.a, &devices[idx], &streams[idx]).unwrap());
            corr_b.push(dtoh_on_stream_sync(&corr.b, &devices[idx], &streams[idx]).unwrap());
        }
        cudarc::nccl::result::group_start().unwrap();
        for (idx, (res, corr)) in izip!(x.iter(), corrections.iter()).enumerate() {
            party.comms()[idx]
                .send_view(&res.b, party.next_id(), &streams[idx])
                .unwrap();
            party.comms()[idx]
                .send_view_u16(&corr.b, party.next_id(), &streams[idx])
                .unwrap();
        }
        for (idx, (res, corr)) in izip!(x.iter_mut(), corrections.iter_mut()).enumerate() {
            party.comms()[idx]
                .receive_view(&mut res.a, party.prev_id(), &streams[idx])
                .unwrap();
            party.comms()[idx]
                .receive_view_u16(&mut corr.a, party.prev_id(), &streams[idx])
                .unwrap();
        }
        cudarc::nccl::result::group_end().unwrap();
        for (idx, (res, corr)) in izip!(x, corrections).enumerate() {
            res_c.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
            corr_c.push(dtoh_on_stream_sync(&corr.a, &devices[idx], &streams[idx]).unwrap());
        }

        let mut result = Vec::with_capacity(n_devices * INPUTS_PER_GPU_SIZE);
        for (mut res_a, res_b, res_c, corr_a, corr_b, corr_c) in
            izip!(res_a, res_b, res_c, corr_a, corr_b, corr_c)
        {
            assert_eq!(res_a.len(), INPUTS_PER_GPU_SIZE);
            assert_eq!(res_b.len(), INPUTS_PER_GPU_SIZE);
            assert_eq!(res_c.len(), INPUTS_PER_GPU_SIZE);
            assert_eq!(corr_a.len(), INPUTS_PER_GPU_SIZE * 2);
            assert_eq!(corr_b.len(), INPUTS_PER_GPU_SIZE * 2);
            assert_eq!(corr_c.len(), INPUTS_PER_GPU_SIZE * 2);

            for (res_a, res_b, res_c, corr_a1, corr_b1, corr_c1, corr_a2, corr_b2, corr_c2) in izip!(
                &mut res_a,
                res_b,
                res_c,
                corr_a.iter().take(INPUTS_PER_GPU_SIZE),
                corr_b.iter().take(INPUTS_PER_GPU_SIZE),
                corr_c.iter().take(INPUTS_PER_GPU_SIZE),
                corr_a.iter().skip(INPUTS_PER_GPU_SIZE),
                corr_b.iter().skip(INPUTS_PER_GPU_SIZE),
                corr_c.iter().skip(INPUTS_PER_GPU_SIZE),
            ) {
                let corr1 = corr_a1.wrapping_add(*corr_b1).wrapping_add(*corr_c1);
                let corr2 = corr_a2.wrapping_add(*corr_b2).wrapping_add(*corr_c2);
                assert!(corr1 == 0 || corr1 == 1);
                assert!(corr2 == 0 || corr2 == 1);
                let mut res = res_a.wrapping_add(res_b).wrapping_add(res_c);
                res -= (corr1 as u32) << 16;
                res -= (corr2 as u32) << 17;
                *res_a = res;
            }
            result.extend(res_a);
        }

        assert_eq!(result.len(), n_devices * INPUTS_PER_GPU_SIZE);
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

        // Import to GPU
        let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices, &streams);
        tracing::info!("id = {}, Data is on GPUs!", id);
        tracing::info!("id = {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            // Simulate Masks to be zero for this test
            let x_ = party.allocate_buffer::<u32>(INPUTS_PER_GPU_SIZE);
            let mut x = to_view(&x_);
            let correction_ = party.allocate_buffer::<u16>(INPUTS_PER_GPU_SIZE * 2);
            let mut correction = to_view(&correction_);
            let mask_gpu = mask_gpu.iter().map(|x| x.as_view()).collect_vec();
            party.synchronize_streams(&streams);

            let now = Instant::now();
            party.lift_mpc(&mask_gpu, &mut x, &mut correction, &streams);
            tracing::info!("id = {}, compute time: {:?}", id, now.elapsed());

            let now = Instant::now();
            let result = open(&mut party, &mut x, &mut correction, &streams);
            party.synchronize_streams(&streams);
            tracing::info!(
                "id = {}, Open and transfer to CPU time: {:?}",
                id,
                now.elapsed()
            );

            let mut correct = true;
            for (i, (r, r_)) in izip!(&result, &real_result).enumerate() {
                if r != r_ {
                    correct = false;
                    tracing::error!("id = {}, Test failed on index: {}: {} != {}", id, i, r, r_);
                    error = true;
                    break;
                }
            }
            if correct {
                tracing::info!("id = {}, Test passed!", id);
            }
        }
        assert!(!error);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn test_lift() -> Result<()> {
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
        let mask_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (mask_share_a, mask_share_b, mask_share_c) = rep_share_vec(&mask_dots, &mut rng);
        let real_result = real_result_msb(mask_dots);
        tracing::info!("Random shared inputs generated!");

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

            testcase(party, mask_share_a, mask_share_c, real_result);
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

            testcase(party, mask_share_b, mask_share_a_, real_result_);
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

            testcase(party, mask_share_c_, mask_share_b_, real_result__);
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
