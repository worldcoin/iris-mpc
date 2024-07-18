use cudarc::driver::{CudaDevice, CudaStream};
use gpu_iris_mpc::{
    helpers::{
        device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync,
        task_monitor::TaskMonitor,
    },
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

// ceil(930 * 125_000 / 2048) * 2048
// const INPUTS_PER_GPU_SIZE: usize = 116_250_624;
const INPUTS_PER_GPU_SIZE: usize = 12_507_136;
const CHUNK_SIZE: usize = INPUTS_PER_GPU_SIZE / 64;

fn rep_share_bin<R: Rng>(value: u64, id: usize, rng: &mut R) -> (u64, u64) {
    let a = rng.gen();
    let b = rng.gen();
    let c = a ^ b ^ value;

    match id {
        0 => (a, c),
        1 => (b, a),
        2 => (c, b),
        _ => unreachable!(),
    }
}

fn rep_share_vec_bin<R: Rng>(value: &[u64], id: usize, rng: &mut R) -> (Vec<u64>, Vec<u64>) {
    let mut a = Vec::with_capacity(value.len());
    let mut b = Vec::with_capacity(value.len());
    for v in value.iter() {
        let (a_, b_) = rep_share_bin(*v, id, rng);
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

    for (dev, stream, a, b) in izip!(devices, streams, a.chunks(CHUNK_SIZE), b.chunks(CHUNK_SIZE)) {
        let a_ = htod_on_stream_sync(a, dev, stream).unwrap();
        let b_ = htod_on_stream_sync(b, dev, stream).unwrap();
        result.push(ChunkShare::new(a_, b_));
    }

    result
}

fn open(party: &mut Circuits, result: &mut ChunkShare<u64>, streams: &[CudaStream]) -> bool {
    let res = result.get_offset(0, 1);
    let mut res_helper = result.get_offset(1, 1);
    cudarc::nccl::result::group_start().expect("group start should work");
    party
        .send_view(&res.b, party.next_id(), 0, streams)
        .unwrap();
    party
        .receive_view(&mut res_helper.a, party.prev_id(), 0, streams)
        .unwrap();
    cudarc::nccl::result::group_end().expect("group end should work");

    let dev = party.get_devices()[0].clone();
    let stream = &streams[0];

    let a = dtoh_on_stream_sync(&res.a, &dev, stream).unwrap();
    let b = dtoh_on_stream_sync(&res.b, &dev, stream).unwrap();
    let c = dtoh_on_stream_sync(&res_helper.a, &dev, stream).unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(c.len(), 1);
    assert!(a[0] == 0 || a[0] == 1);
    assert!(b[0] == 0 || b[0] == 1);
    assert!(c[0] == 0 || c[0] == 1);
    let result = a[0] ^ b[0] ^ c[0];
    result == 1
}

#[tokio::test]
#[ignore]
async fn main() -> eyre::Result<()> {
    assert!(
        INPUTS_PER_GPU_SIZE % (2048) == 0,
        // Mod 16 for randomness, mod 64 for chunk size
        "Inputs per GPU size must be a multiple of 2048"
    );
    // TODO
    let mut rng = StdRng::seed_from_u64(42);

    let party_id: usize = env::var("PARTY_ID")
        .expect("PARTY_ID environment variable not set")
        .parse()
        .expect("PARTY_ID must be a valid usize");
    let url = env::var("PEER_URL")?;
    let n_devices = CudaDevice::count()? as usize;

    // Get Circuit Party
    let device_manager = Arc::new(DeviceManager::init());
    let mut server_tasks = TaskMonitor::new();
    let mut party = Circuits::new(
        party_id,
        INPUTS_PER_GPU_SIZE,
        INPUTS_PER_GPU_SIZE / 64,
        ([party_id as u32; 8], [((party_id + 2) % 3) as u32; 8]),
        Some(url),
        Some(9001),
        Some(&mut server_tasks),
        device_manager.clone(),
    );
    let devices = party.get_devices();
    let streams = devices
        .iter()
        .map(|dev| dev.fork_default_stream().unwrap())
        .collect::<Vec<_>>();
    server_tasks.check_tasks();

    println!("Starting tests...");
    for i in 0..=n_devices {
        server_tasks.check_tasks();

        println!("Test: {}", i);
        let mut inputs = vec![0; CHUNK_SIZE * n_devices];
        if i < n_devices {
            let index = rng.gen_range(i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE);
            let word_index = rng.gen_range(0..64);
            inputs[index] = 1 << word_index;
        }

        let (share_a, share_b) = rep_share_vec_bin(&inputs, party_id, &mut rng);

        // Import to GPU
        let mut share_gpu = to_gpu(&share_a, &share_b, &devices, &streams);
        println!("Data is on GPUs!");

        let now = Instant::now();
        party.or_reduce_result(&mut share_gpu, &streams);
        println!("compute time: {:?}", now.elapsed());

        let now = Instant::now();
        // Result is in the first bit of the first GPU
        let result = open(&mut party, &mut share_gpu[0], &streams);
        party.synchronize_streams(&streams);
        println!("Open and transfer to CPU time: {:?}", now.elapsed());

        if i == n_devices {
            if result {
                println!("Test failed!");
            } else {
                println!("Test passed!");
            }
        } else if result {
            println!("Test passed!");
        } else {
            println!("Test failed!");
        }
    }

    server_tasks.abort_all();
    time::sleep(time::Duration::from_secs(5)).await;
    server_tasks.check_tasks_finished();
    Ok(())
}
