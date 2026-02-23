#[cfg(feature = "gpu_dependent")]
mod extract_msb_mod_test {

    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
    use eyre::Result;
    use iris_mpc_common::iris_db::iris::IrisCodeArray;
    use iris_mpc_gpu::{
        dot::THRESHOLD_A,
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
    const CHUNK_SIZE: usize = INPUTS_PER_GPU_SIZE / 64;
    const B_BITS: u64 = 16;

    fn to_view<'a, T>(inp: &'a [ChunkShare<T>]) -> Vec<ChunkShareView<'a, T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.as_view());
        }
        res
    }

    fn sample_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
        (0..size)
            .map(|_| {
                let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
                let neg = rng.gen::<bool>();
                if neg {
                    x = (u16::MAX - x).wrapping_add(1);
                }
                x
            })
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

    fn pack_with_device_padding(bits: Vec<bool>) -> Vec<u64> {
        assert!(bits.len() % INPUTS_PER_GPU_SIZE == 0);
        let mut res = vec![];
        for devices in bits.chunks_exact(INPUTS_PER_GPU_SIZE) {
            for bits in devices.chunks(64) {
                let mut r = 0;
                for (i, bit) in bits.iter().enumerate() {
                    r |= u64::from(*bit) << i;
                }
                res.push(r);
            }
        }
        res
    }

    #[allow(clippy::precedence)]
    fn real_result_msb(input: Vec<u16>) -> Vec<u64> {
        let mod_ = 1u64 << (16 + B_BITS);
        let mut res = Vec::with_capacity(input.len());
        for inp in input {
            let r = (u64::MAX - ((inp as u64) << B_BITS)) % mod_;
            let msb = r >> (B_BITS + 16 - 1) & 1 == 1;
            res.push(msb)
        }
        pack_with_device_padding(res)
    }

    fn open(party: &mut Circuits, x: &[ChunkShare<u64>], streams: &[CudaStream]) -> Vec<u64> {
        let n_devices = x.len();
        let mut a = Vec::with_capacity(n_devices);
        let mut b = Vec::with_capacity(n_devices);
        let mut c = Vec::with_capacity(n_devices);

        cudarc::nccl::result::group_start().unwrap();
        for (idx, res) in x.iter().enumerate() {
            // Result is in bit 0
            let res = res.get_offset(0, CHUNK_SIZE);
            party.comms()[idx]
                .send_view(&res.b, party.next_id(), &streams[idx])
                .unwrap();
            a.push(res.a);
            b.push(res.b);
        }
        for (idx, res) in x.iter().enumerate() {
            let mut res = res.get_offset(1, CHUNK_SIZE);
            party.comms()[idx]
                .receive_view(&mut res.a, party.prev_id(), &streams[idx])
                .unwrap();
            c.push(res.a);
        }
        cudarc::nccl::result::group_end().unwrap();

        let mut result = Vec::with_capacity(n_devices * CHUNK_SIZE);
        let devices = party.get_devices();
        for (dev, stream, a, b, c) in izip!(devices, streams, a, b, c) {
            let mut a = dtoh_on_stream_sync(&a, &dev, stream).unwrap();
            let b = dtoh_on_stream_sync(&b, &dev, stream).unwrap();
            let c = dtoh_on_stream_sync(&c, &dev, stream).unwrap();
            for (a, b, c) in izip!(a.iter_mut(), b, c) {
                *a ^= b ^ c;
            }
            result.extend(a);
        }
        assert_eq!(result.len(), n_devices * CHUNK_SIZE);
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
        real_result: Vec<u64>,
    ) {
        let id = party.peer_id();

        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        // Import to GPU
        let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices, &streams);
        tracing::info!("id = {}, Data is on GPUs!", id);
        tracing::info!("id = {}, Starting tests...", id);

        let mut error = false;
        for _ in 0..10 {
            // Simulate Masks to be zero for this test
            let x_ = party.allocate_buffer::<u32>(INPUTS_PER_GPU_SIZE);
            let mut x = to_view(&x_);
            let correction_ = party.allocate_buffer::<u16>(INPUTS_PER_GPU_SIZE * 2);
            let correction = to_view(&correction_);
            let code_gpu = code_gpu.iter().map(|x| x.as_view()).collect_vec();
            party.synchronize_streams(&streams);

            let now = Instant::now();
            party.lift_mul_sub(&mut x, &correction, &code_gpu, THRESHOLD_A, &streams);
            tracing::info!("id = {}, lift time: {:?}", id, now.elapsed());
            party.extract_msb(&mut x, &streams);
            tracing::info!("id = {}, extract time: {:?}", id, now.elapsed());

            let res = party.take_result_buffer();
            let now = Instant::now();
            let result = open(&mut party, &res, &streams);
            party.synchronize_streams(&streams);
            party.return_result_buffer(res);
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
    async fn test_extract_msb_mod() -> Result<()> {
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
        let code_dots = sample_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
        let (code_share_a, code_share_b, code_share_c) = rep_share_vec(&code_dots, &mut rng);
        let real_result = real_result_msb(code_dots);
        tracing::info!("Random shared inputs generated!");

        let code_share_a_ = code_share_a.to_owned();
        let code_share_b_ = code_share_b.to_owned();
        let code_share_c_ = code_share_c.to_owned();
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

            testcase(party, code_share_a, code_share_c, real_result);
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

            testcase(party, code_share_b, code_share_a_, real_result_);
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

            testcase(party, code_share_c_, code_share_b_, real_result__);
        });

        task0.await.unwrap();
        task1.await.unwrap();
        task2.await.unwrap();

        Ok(())
    }
}
