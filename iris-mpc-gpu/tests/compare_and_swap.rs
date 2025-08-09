#[cfg(feature = "gpu_dependent")]
mod compare_and_swap_test {
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
    const INPUTS_PER_GPU_SIZE: usize = 64 * 1024;

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

    #[allow(clippy::precedence)]
    fn real_result(
        code_input: Vec<u32>,
        mask_input: Vec<u32>,
        code_input_2: Vec<u32>,
        mask_input_2: Vec<u32>,
    ) -> (Vec<u32>, Vec<u32>) {
        assert_eq!(code_input.len(), mask_input.len());

        let mut code = Vec::with_capacity(code_input.len());
        let mut mask = Vec::with_capacity(mask_input.len());
        for (code_1, mask_1, code_2, mask_2) in
            izip!(code_input, mask_input, code_input_2, mask_input_2)
        {
            let one_smaller_than_two = code_2 * mask_1 < code_1 * mask_2;
            let (code_res, mask_res) = if one_smaller_than_two {
                (code_1, mask_1)
            } else {
                (code_2, mask_2)
            };
            code.push(code_res);
            mask.push(mask_res);
        }
        (code, mask)
    }

    fn open(
        party: &mut Circuits,
        x: &mut [ChunkShareView<u32>],
        streams: &[CudaStream],
    ) -> Vec<u32> {
        let n_devices = x.len();
        let mut res_a = Vec::with_capacity(n_devices);
        let mut res_b = Vec::with_capacity(n_devices);
        let mut res_c = Vec::with_capacity(n_devices);

        let devices = party.get_devices();
        for (idx, res) in x.iter().enumerate() {
            res_a.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
            res_b.push(dtoh_on_stream_sync(&res.b, &devices[idx], &streams[idx]).unwrap());
        }
        cudarc::nccl::result::group_start().unwrap();
        for (idx, res) in x.iter().enumerate() {
            party.comms()[idx]
                .send_view(&res.b, party.next_id(), &streams[idx])
                .unwrap();
        }
        for (idx, res) in x.iter_mut().enumerate() {
            party.comms()[idx]
                .receive_view(&mut res.a, party.prev_id(), &streams[idx])
                .unwrap();
        }
        cudarc::nccl::result::group_end().unwrap();
        for (idx, res) in x.iter().enumerate() {
            res_c.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
        }

        let mut result = Vec::with_capacity(n_devices * INPUTS_PER_GPU_SIZE);
        for (mut res_a, res_b, res_c) in izip!(res_a, res_b, res_c) {
            assert_eq!(res_a.len(), INPUTS_PER_GPU_SIZE);
            assert_eq!(res_b.len(), INPUTS_PER_GPU_SIZE);
            assert_eq!(res_c.len(), INPUTS_PER_GPU_SIZE);

            for (res_a, res_b, res_c) in izip!(&mut res_a, res_b, res_c,) {
                let res = res_a.wrapping_add(res_b).wrapping_add(res_c);
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

    #[allow(clippy::too_many_arguments)]
    fn testcase(
        mut party: Circuits,
        code_share_a: Vec<u32>,
        code_share_b: Vec<u32>,
        mask_share_a: Vec<u32>,
        mask_share_b: Vec<u32>,
        code_2_share_a: Vec<u32>,
        code_2_share_b: Vec<u32>,
        mask_2_share_a: Vec<u32>,
        mask_2_share_b: Vec<u32>,
        real_code_result: Vec<u32>,
        real_mask_result: Vec<u32>,
    ) {
        let id = party.peer_id();

        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        // Import to GPU
        let code_2_gpu = to_gpu(&code_2_share_a, &code_2_share_b, &devices, &streams);
        let mask_2_gpu = to_gpu(&mask_2_share_a, &mask_2_share_b, &devices, &streams);
        tracing::info!("id: {}, Data is on GPUs!", id);
        tracing::info!("id: {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices, &streams);
            let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices, &streams);
            let mut code_gpu = code_gpu.iter().map(|x| x.as_view()).collect_vec();
            let mut mask_gpu = mask_gpu.iter().map(|x| x.as_view()).collect_vec();
            let code_2_gpu = code_2_gpu.iter().map(|x| x.as_view()).collect_vec();
            let mask_2_gpu = mask_2_gpu.iter().map(|x| x.as_view()).collect_vec();
            party.synchronize_streams(&streams);

            let now = Instant::now();
            party.cross_compare_and_swap(
                &mut code_gpu,
                &mut mask_gpu,
                &code_2_gpu,
                &mask_2_gpu,
                &streams,
            );

            party.synchronize_streams(&streams);
            tracing::info!("id: {}, compute time: {:?}", id, now.elapsed());

            let now = Instant::now();
            let result_code = open(&mut party, &mut code_gpu, &streams);
            let result_mask = open(&mut party, &mut mask_gpu, &streams);
            tracing::info!("id: {}, Starting tests...", id);
            tracing::info!(
                "id: {}, Open and transfer to CPU time: {:?}",
                id,
                now.elapsed()
            );

            let mut correct = true;
            for (i, (c, m, rc, rm)) in izip!(
                &result_code,
                &result_mask,
                &real_code_result,
                &real_mask_result
            )
            .enumerate()
            {
                if c != rc || m != rm {
                    correct = false;
                    tracing::error!(
                        "id: {}, Test failed on index: {}: {} != {} or {} != {}",
                        id,
                        i,
                        c,
                        rc,
                        m,
                        rm
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
    async fn test_compare_and_swap() -> Result<()> {
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
        let code2_dots = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let mask2_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (code_share_a, code_share_b, code_share_c) = rep_share_vec(&code_dots, &mut rng);
        let (mask_share_a, mask_share_b, mask_share_c) = rep_share_vec(&mask_dots, &mut rng);
        let (code2_share_a, code2_share_b, code2_share_c) = rep_share_vec(&code2_dots, &mut rng);
        let (mask2_share_a, mask2_share_b, mask2_share_c) = rep_share_vec(&mask2_dots, &mut rng);
        let real_result = real_result(code_dots, mask_dots, code2_dots, mask2_dots);
        tracing::info!("Random shared inputs generated!");

        let code_share_a_ = code_share_a.to_owned();
        let code_share_b_ = code_share_b.to_owned();
        let code_share_c_ = code_share_c.to_owned();
        let mask_share_a_ = mask_share_a.to_owned();
        let mask_share_b_ = mask_share_b.to_owned();
        let mask_share_c_ = mask_share_c.to_owned();
        let code2_share_a_ = code2_share_a.to_owned();
        let code2_share_b_ = code2_share_b.to_owned();
        let code2_share_c_ = code2_share_c.to_owned();
        let mask2_share_a_ = mask2_share_a.to_owned();
        let mask2_share_b_ = mask2_share_b.to_owned();
        let mask2_share_c_ = mask2_share_c.to_owned();
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
                code_share_a,
                code_share_c,
                mask_share_a,
                mask_share_c,
                code2_share_a,
                code2_share_c,
                mask2_share_a,
                mask2_share_c,
                real_result.0,
                real_result.1,
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
                code_share_b,
                code_share_a_,
                mask_share_b,
                mask_share_a_,
                code2_share_b,
                code2_share_a_,
                mask2_share_b,
                mask2_share_a_,
                real_result_.0,
                real_result_.1,
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
                code_share_c_,
                code_share_b_,
                mask_share_c_,
                mask_share_b_,
                code2_share_c_,
                code2_share_b_,
                mask2_share_c_,
                mask2_share_b_,
                real_result__.0,
                real_result__.1,
            );
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
