#[allow(unused)]
mod bitinject_test {
    use cudarc::driver::{CudaDevice, CudaStream};
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
        threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
    };
    use itertools::izip;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use static_assertions::const_assert;
    use std::{env, sync::Arc};
    use tokio::time::Instant;

    const INPUTS_PER_GPU_SIZE: usize = 2048 * 2;

    fn to_view<T>(inp: &[ChunkShare<T>]) -> Vec<ChunkShareView<T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.as_view());
        }
        res
    }

    fn sample_bits<R: Rng>(size: usize, rng: &mut R) -> Vec<u64> {
        (0..size / 64).map(|_| rng.gen()).collect::<Vec<_>>()
    }

    fn rep_share<R: Rng>(value: u64, id: usize, rng: &mut R) -> (u64, u64) {
        let a = rng.next_u64();
        let b = rng.next_u64();
        let c = value ^ a ^ b;

        match id {
            0 => (a, c),
            1 => (b, a),
            2 => (c, b),
            _ => unreachable!(),
        }
    }

    fn rep_share_vec<R: Rng>(value: &[u64], id: usize, rng: &mut R) -> (Vec<u64>, Vec<u64>) {
        let mut a = Vec::with_capacity(value.len());
        let mut b = Vec::with_capacity(value.len());
        for v in value.iter() {
            let (a_, b_) = rep_share(*v, id, rng);
            a.push(a_);
            b.push(b_);
        }
        (a, b)
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
                *a += b + c;
            }
            result.extend(a);
        }
        assert_eq!(result.len(), n_devices * INPUTS_PER_GPU_SIZE);
        result
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[cfg(feature = "gpu_dependent")]
    #[ignore]
    async fn test_bitinject() -> eyre::Result<()> {
        const_assert!(
            INPUTS_PER_GPU_SIZE % (2048) == 0,
            // Mod 16 for randomness, mod 64 for chunk size
        );

        // TODO
        let mut rng = StdRng::seed_from_u64(42);

        let party_id: usize = env::var("PARTY_ID")
            .expect("PARTY_ID environment variable not set")
            .parse()
            .expect("PARTY_ID must be a valid usize");
        let n_devices = CudaDevice::count()? as usize;

        // Get inputs
        let input_bits = sample_bits(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

        let (input_bits_a, input_bits_b) = rep_share_vec(&input_bits, party_id, &mut rng);
        let real_result = real_result(input_bits);
        println!("Random shared inputs generated!");

        // Get Circuit Party
        let device_manager = Arc::new(DeviceManager::init());
        let ids = device_manager.get_ids_from_magic(0);
        let comms = device_manager.instantiate_network_from_ids(party_id, &ids)?;
        let mut party = Circuits::new(
            party_id,
            INPUTS_PER_GPU_SIZE / 2,
            INPUTS_PER_GPU_SIZE / 128,
            ([party_id as u32; 8], [((party_id + 2) % 3) as u32; 8]),
            device_manager.clone(),
            comms,
        );
        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        // Import to GPU
        let code_gpu = to_gpu(&input_bits_a, &input_bits_b, &devices, &streams);
        let res_ = alloc_res(INPUTS_PER_GPU_SIZE, &devices);
        let mut res = to_view(&res_);
        println!("Data is on GPUs!");
        println!("Starting tests...");

        for _ in 0..10 {
            let code_gpu_ = code_gpu.clone();
            let code_gpu = to_view(&code_gpu_);

            let now = Instant::now();
            party.bit_inject_ot(&code_gpu, &mut res, &streams);
            println!("compute time: {:?}", now.elapsed());

            let now = Instant::now();
            let result = open(&mut party, &mut res, &streams);
            println!("Open and transfer to CPU time: {:?}", now.elapsed());
            party.synchronize_streams(&streams);
            let mut correct = true;
            for (i, (r, r_)) in izip!(&result, &real_result).enumerate() {
                if r != r_ {
                    correct = false;
                    println!("Test failed on index: {}: {} != {}", i, r, r_);
                    break;
                }
            }
            if correct {
                println!("Test passed!");
            }
        }

        Ok(())
    }
}
