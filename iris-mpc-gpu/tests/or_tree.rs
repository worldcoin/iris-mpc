#[cfg(feature = "gpu_dependent")]
mod or_tree_test {
    use cudarc::{
        driver::{CudaDevice, CudaStream},
        nccl::Id,
    };
    use iris_mpc_gpu::{
        helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
        threshold_ring::protocol::{ChunkShare, Circuits},
    };
    use itertools::izip;
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

    fn rep_share_bin<R: Rng>(value: u64, rng: &mut R) -> (u64, u64, u64) {
        let a = rng.gen();
        let b = rng.gen();
        let c = a ^ b ^ value;

        (a, b, c)
    }

    fn rep_share_vec_bin<R: Rng>(value: &[u64], rng: &mut R) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut a = Vec::with_capacity(value.len());
        let mut b = Vec::with_capacity(value.len());
        let mut c = Vec::with_capacity(value.len());
        for v in value.iter() {
            let (a_, b_, c_) = rep_share_bin(*v, rng);
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

        for (dev, stream, a, b) in
            izip!(devices, streams, a.chunks(CHUNK_SIZE), b.chunks(CHUNK_SIZE))
        {
            let a_ = htod_on_stream_sync(a, dev, stream).unwrap();
            let b_ = htod_on_stream_sync(b, dev, stream).unwrap();
            result.push(ChunkShare::new(a_, b_));
        }

        result
    }

    fn open(party: &mut Circuits, result: &mut ChunkShare<u64>, streams: &[CudaStream]) -> bool {
        let dev = party.get_devices()[0].clone();
        let stream = &streams[0];

        let mut res = result.get_offset(0, 1);
        let a = dtoh_on_stream_sync(&res.a, &dev, stream).unwrap();
        let b = dtoh_on_stream_sync(&res.b, &dev, stream).unwrap();

        cudarc::nccl::result::group_start().expect("group start should work");
        party.comms()[0]
            .send_view(&res.b, party.next_id(), stream)
            .unwrap();
        party.comms()[0]
            .receive_view(&mut res.a, party.prev_id(), stream)
            .unwrap();
        cudarc::nccl::result::group_end().expect("group end should work");

        let c = dtoh_on_stream_sync(&res.a, &dev, stream).unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(c.len(), 1);
        assert!(a[0] == 0 || a[0] == 1);
        assert!(b[0] == 0 || b[0] == 1);
        assert!(c[0] == 0 || c[0] == 1);
        let result = a[0] ^ b[0] ^ c[0];
        result == 1
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

    fn testcase(mut party: Circuits, share_a: Vec<u64>, share_b: Vec<u64>, i: usize) {
        let id = party.peer_id();

        let devices = party.get_devices();
        let streams = devices
            .iter()
            .map(|dev| dev.fork_default_stream().unwrap())
            .collect::<Vec<_>>();

        // Import to GPU
        let mut share_gpu = to_gpu(&share_a, &share_b, &devices, &streams);
        party.synchronize_streams(&streams);
        tracing::info!("id = {}, Data is on GPUs!", id);

        let now = Instant::now();
        party.or_reduce_result(&mut share_gpu, &streams);
        tracing::info!("id = {}, compute time: {:?}", id, now.elapsed());

        let now = Instant::now();
        // Result is in the first bit of the first GPU
        let result = open(&mut party, &mut share_gpu[0], &streams);
        party.synchronize_streams(&streams);
        tracing::info!(
            "id = {}, Open and transfer to CPU time: {:?}",
            id,
            now.elapsed()
        );

        if i == devices.len() {
            if result {
                tracing::error!("id = {}, Test failed!", id);
                panic!();
            } else {
                tracing::info!("id = {}, Test passed!", id);
            }
        } else if result {
            tracing::info!("id = {}, Test passed!", id);
        } else {
            tracing::error!("id = {}, Test failed!", id);
            panic!();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn main() -> eyre::Result<()> {
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
        let device_manager2_ = Arc::new(device_managers.pop().unwrap());
        let device_manager1_ = Arc::new(device_managers.pop().unwrap());
        let device_manager0_ = Arc::new(device_managers.pop().unwrap());
        let n_devices = device_manager0_.devices().len();
        let ids0_ = (0..n_devices)
            .map(|_| Id::new().unwrap())
            .collect::<Vec<_>>();

        tracing::info!("Starting tests...");
        for i in 0..=n_devices {
            let ids0 = ids0_.clone();
            let ids1 = ids0.clone();
            let ids2 = ids0.clone();
            let device_manager0 = device_manager0_.clone();
            let device_manager1 = device_manager1_.clone();
            let device_manager2 = device_manager2_.clone();

            tracing::info!("Test: {}", i);
            let mut inputs = vec![0; CHUNK_SIZE * n_devices];
            if i < n_devices {
                let index = rng.gen_range(i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE);
                let word_index = rng.gen_range(0..64);
                inputs[index] = 1 << word_index;
            }

            let (share_a, share_b, share_c) = rep_share_vec_bin(&inputs, &mut rng);

            let share_a_ = share_a.to_owned();
            let share_b_ = share_b.to_owned();
            let share_c_ = share_c.to_owned();

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

                testcase(party, share_a, share_c, i);
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

                testcase(party, share_b, share_a_, i);
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

                testcase(party, share_c_, share_b_, i);
            });

            task0.await.unwrap();
            task1.await.unwrap();
            task2.await.unwrap();
        }

        Ok(())
    }
}
