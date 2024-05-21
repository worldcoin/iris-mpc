use cudarc::driver::CudaDevice;
use gpu_iris_mpc::threshold::protocol::{ChunkShare, Circuits};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

// const INPUTS_PER_GPU_SIZE: usize = 2 * 116_250_048; //ceil(930 * 125_000 / 64) * 64
const INPUTS_PER_GPU_SIZE: usize = 12_505_600;
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

fn to_gpu(a: &[u64], b: &[u64], devices: &[Arc<CudaDevice>]) -> Vec<ChunkShare<u64>> {
    debug_assert_eq!(a.len(), b.len());

    let mut result = Vec::with_capacity(devices.len());

    for (dev, a, b) in izip!(devices, a.chunks(CHUNK_SIZE), b.chunks(CHUNK_SIZE)) {
        let a_ = dev.htod_sync_copy(a).unwrap();
        let b_ = dev.htod_sync_copy(b).unwrap();
        result.push(ChunkShare::new(a_, b_));
    }

    result
}

fn open(party: &mut Circuits, result: &mut ChunkShare<u64>) -> bool {
    let res = result.get_offset(0, 1);
    let mut res_helper = result.get_offset(1, 1);
    cudarc::nccl::result::group_start().expect("group start should work");
    party.send_view(&res.b, party.next_id(), 0);
    party.receive_view(&mut res_helper.a, party.prev_id(), 0);
    cudarc::nccl::result::group_end().expect("group end should work");

    let dev = party.get_devices()[0].clone();

    let a = dev.dtoh_sync_copy(&res.a).expect("copy a works");
    let b = dev.dtoh_sync_copy(&res.b).expect("copy b works");
    let c = dev.dtoh_sync_copy(&res_helper.a).expect("copy c works");
    assert_eq!(a.len(), 1);
    assert_eq!(b.len(), 1);
    assert_eq!(c.len(), 1);
    assert!(a[0] == 0 || a[0] == 1);
    assert!(b[0] == 0 || b[0] == 1);
    assert!(c[0] == 0 || c[0] == 1);
    let result = a[0] ^ b[0] ^ c[0];
    result == 1
}

#[allow(clippy::assertions_on_constants)]
#[tokio::main(worker_threads = 1)]
async fn main() -> eyre::Result<()> {
    assert!(
        INPUTS_PER_GPU_SIZE % 64 == 0,
        "Inputs per GPU size must be a multiple of 64"
    );
    // TODO
    let mut rng = StdRng::seed_from_u64(42);

    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);
    let n_devices = CudaDevice::count().unwrap() as usize;

    // Get Circuit Party
    let mut party = Circuits::new(party_id, INPUTS_PER_GPU_SIZE, url, Some(3001));
    let devices = party.get_devices();

    println!("Starting tests...");
    for i in 0..=n_devices {
        println!("Test: {}", i);
        let mut inputs = vec![0; CHUNK_SIZE * n_devices];
        if i < n_devices {
            let index = rng.gen_range(i * CHUNK_SIZE..(i + 1) * CHUNK_SIZE);
            let word_index = rng.gen_range(0..64);
            inputs[index] = 1 << word_index;
        }

        let (share_a, share_b) = rep_share_vec_bin(&inputs, party_id, &mut rng);

        // Import to GPU
        let mut share_gpu = to_gpu(&share_a, &share_b, &devices);
        println!("Data is on GPUs!");

        let now = Instant::now();
        party.or_reduce_result(&mut share_gpu);
        party.synchronize_all();
        println!("compute time: {:?}", now.elapsed());

        let now = Instant::now();
        // Result is in the first bit of the first GPU
        let result = open(&mut party, &mut share_gpu[0]);
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

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
