#[cfg(feature = "gpu_dependent")]
mod bitinject_test {
    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
    use eyre::Result;
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
        threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
    };
    use itertools::izip;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use static_assertions::const_assert;
    use std::{env, sync::Arc};
    use tokio::time::Instant;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    const DB_RNG_SEED: u64 = 0xdeadbeef;
    const INPUTS_PER_GPU_SIZE: usize = 2048 * 2;

    fn to_view<'a, T>(inp: &'a [ChunkShare<T>]) -> Vec<ChunkShareView<'a, T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.as_view());
        }
        res
    }

    fn sample_bits<R: Rng>(size: usize, rng: &mut R) -> Vec<u64> {
        (0..size / 64).map(|_| rng.gen()).collect::<Vec<_>>()
    }

    fn rep_share<R: Rng>(value: u64, rng: &mut R) -> (u64, u64, u64) {
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = value ^ a ^ b;

        (a, b, c)
    }

    fn rep_share_vec<R: Rng>(value: &[u64], rng: &mut R) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
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
        a: &[u64],
        b: &[u64],
        devices: &[Arc<CudaDevice>],
        streams: &[CudaStream],
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(a.len(), b.len());

        let mut result = Vec::with_capacity(devices.len());

        for (dev, stream, a, b) in izip!(
            devices,
            streams,
            a.chunks(INPUTS_PER_GPU_SIZE / 64),
            b.chunks(INPUTS_PER_GPU_SIZE / 64)
        ) {
            let a_ = htod_on_stream_sync(a, dev, stream).unwrap();
            let b_ = htod_on_stream_sync(b, dev, stream).unwrap();
            result.push(ChunkShare::new(a_, b_));
        }

        result
    }

    fn alloc_res(size: usize, devices: &[Arc<CudaDevice>]) -> Vec<ChunkShare<u16>> {
        devices
            .iter()
            .map(|dev| {
                let a = dev.alloc_zeros(size).unwrap();
                let b = dev.alloc_zeros(size).unwrap();
                ChunkShare::new(a, b)
            })
            .collect()
    }

    fn real_result(input: Vec<u64>) -> Vec<u16> {
        let mut res = Vec::with_capacity(input.len());
        for i in input.into_iter() {
            for j in 0..64 {
                res.push(((i >> j) & 1) as u16)
            }
        }
        res
    }

    fn open(
        party: &mut Circuits,
        x: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) -> Vec<u16> {
        let n_devices = x.len();
        let mut a = Vec::with_capacity(n_devices);
        let mut b = Vec::with_capacity(n_devices);
        let mut c = Vec::with_capacity(n_devices);

        let devices = party.get_devices();
        for (idx, res) in x.iter().enumerate() {
            a.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
            b.push(dtoh_on_stream_sync(&res.b, &devices[idx], &streams[idx]).unwrap());
        }
        cudarc::nccl::result::group_start().unwrap();
        for (idx, res) in x.iter().enumerate() {
            party.comms()[idx]
                .send_view_u16(&res.b, party.next_id(), &streams[idx])
                .unwrap();
        }
        for (idx, res) in x.iter_mut().enumerate() {
            party.comms()[idx]
                .receive_view_u16(&mut res.a, party.prev_id(), &streams[idx])
                .unwrap();
        }
        cudarc::nccl::result::group_end().unwrap();
        for (idx, res) in x.iter_mut().enumerate() {
            c.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap())
        }

        let mut result = Vec::with_capacity(n_devices * INPUTS_PER_GPU_SIZE);
        for (mut a, b, c) in izip!(a, b, c) {
            for (a, b, c) in izip!(a.iter_mut(), b, c) {
                *a = a.wrapping_add(b).wrapping_add(c);
            }
            result.extend(a);
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
        input_bits_a: Vec<u64>,
        input_bits_b: Vec<u64>,
        real_result: Vec<u16>,
    ) {
        let id = party.peer_id();

        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        // Import to GPU
        let code_gpu = to_gpu(&input_bits_a, &input_bits_b, &devices, &streams);
        let res_ = alloc_res(INPUTS_PER_GPU_SIZE, &devices);
        let mut res = to_view(&res_);
        tracing::info!("id = {}, Data is on GPUs!", id);
        tracing::info!("id = {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            let code_gpu_ = code_gpu.clone();
            let code_gpu = to_view(&code_gpu_);
            party.synchronize_streams(&streams);

            let now = Instant::now();
            party.bit_inject_ot(&code_gpu, &mut res, &streams);
            tracing::info!("id = {}, compute time: {:?}", id, now.elapsed());

            let now = Instant::now();
            let result = open(&mut party, &mut res, &streams);
            tracing::info!(
                "id = {}, Open and transfer to CPU time: {:?}",
                id,
                now.elapsed()
            );
            party.synchronize_streams(&streams);
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
    async fn test_bitinject() -> Result<()> {
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
        let input_bits = sample_bits(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (input_bits_a, input_bits_b, input_bits_c) = rep_share_vec(&input_bits, &mut rng);
        let real_result = real_result(input_bits);
        tracing::info!("Random shared inputs generated!");

        let input_bits_a_ = input_bits_a.to_owned();
        let input_bits_b_ = input_bits_b.to_owned();
        let input_bits_c_ = input_bits_c.to_owned();
        let real_result_ = real_result.to_owned();
        let real_result__ = real_result.to_owned();

        let task0 = tokio::task::spawn_blocking(move || {
            let comms0 = device_manager0
                .instantiate_network_from_ids(0, &ids0)
                .unwrap();

            let party = Circuits::new(
                0,
                INPUTS_PER_GPU_SIZE / 2,
                INPUTS_PER_GPU_SIZE / 128,
                chacha_seeds0,
                device_manager0,
                comms0,
            );

            testcase(party, input_bits_a, input_bits_c, real_result);
        });

        let task1 = tokio::task::spawn_blocking(move || {
            let comms1 = device_manager1
                .instantiate_network_from_ids(1, &ids1)
                .unwrap();

            let party = Circuits::new(
                1,
                INPUTS_PER_GPU_SIZE / 2,
                INPUTS_PER_GPU_SIZE / 128,
                chacha_seeds1,
                device_manager1,
                comms1,
            );

            testcase(party, input_bits_b, input_bits_a_, real_result_);
        });

        let task2 = tokio::task::spawn_blocking(move || {
            let comms2 = device_manager2
                .instantiate_network_from_ids(2, &ids2)
                .unwrap();

            let party = Circuits::new(
                2,
                INPUTS_PER_GPU_SIZE / 2,
                INPUTS_PER_GPU_SIZE / 128,
                chacha_seeds2,
                device_manager2,
                comms2,
            );

            testcase(party, input_bits_c_, input_bits_b_, real_result__);
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();
        Ok(())
    }
}
