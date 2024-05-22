use cudarc::driver::CudaDevice;
use gpu_iris_mpc::threshold::protocol::{ChunkShare, Circuits};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

const INPUTS_PER_GPU_SIZE: usize = 2048;

fn sample_bits<R: Rng>(size: usize, rng: &mut R) -> Vec<u64> {
    (0..size / 64).map(|_| rng.gen::<u64>()).collect::<Vec<_>>()
}

fn rep_share<R: Rng>(value: u64, id: usize, rng: &mut R) -> (u64, u64) {
    let a = rng.next_u64();
    let b = rng.next_u64();
    let c = value - a - b;

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

fn to_gpu(a: &[u64], b: &[u64], devices: &[Arc<CudaDevice>]) -> Vec<ChunkShare<u64>> {
    debug_assert_eq!(a.len(), b.len());

    let mut result = Vec::with_capacity(devices.len());

    for (dev, a, b) in izip!(
        devices,
        a.chunks(INPUTS_PER_GPU_SIZE),
        b.chunks(INPUTS_PER_GPU_SIZE)
    ) {
        let a_ = dev.htod_sync_copy(a).unwrap();
        let b_ = dev.htod_sync_copy(b).unwrap();
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
    let mut res = Vec::with_capacity(input.len() * 64);
    for i in input.into_iter() {
        for j in 0..64 {
            res.push(1 - ((i >> j) & 1) as u16)
        }
    }
    res
}

fn open(party: &mut Circuits, x: &mut [ChunkShare<u16>]) -> Vec<u16> {
    let n_devices = x.len();
    let mut a = Vec::with_capacity(n_devices);
    let mut b = Vec::with_capacity(n_devices);
    let mut c = Vec::with_capacity(n_devices);

    let devices = party.get_devices();
    cudarc::nccl::result::group_start().unwrap();
    for (idx, res) in x.iter().enumerate() {
        // Result is in bit 0
        party.send_u16(&res.b, party.next_id(), idx);
        a.push(devices[idx].dtoh_sync_copy(&res.a).unwrap());
        b.push(devices[idx].dtoh_sync_copy(&res.b).unwrap());
    }
    for (idx, res) in x.iter_mut().enumerate() {
        party.receive_u16(&mut res.a, party.prev_id(), idx);
        c.push(devices[idx].dtoh_sync_copy(&res.a).unwrap());
    }
    cudarc::nccl::result::group_end().unwrap();

    let mut result = Vec::with_capacity(n_devices * INPUTS_PER_GPU_SIZE);
    for (mut a, b, c) in izip!(a, b, c) {
        for (a, b, c) in izip!(a.iter_mut(), b, c) {
            *a ^= b ^ c;
        }
        result.extend(a);
    }
    assert_eq!(result.len(), n_devices * INPUTS_PER_GPU_SIZE);
    result
}

#[allow(clippy::assertions_on_constants)]
#[tokio::main(worker_threads = 1)]
async fn main() -> eyre::Result<()> {
    assert!(
        INPUTS_PER_GPU_SIZE % (2048) == 0,
        // Mod 16 for randomness, mod 64 for chunk size
        "Inputs per GPU size must be a multiple of 2048"
    );
    // TODO
    let mut rng = StdRng::seed_from_u64(42);

    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);
    let n_devices = CudaDevice::count().unwrap() as usize;

    // Get inputs
    let input_bits = sample_bits(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

    let (input_bits_a, input_bits_b) = rep_share_vec(&input_bits, party_id, &mut rng);
    let real_result = real_result(input_bits);
    println!("Random shared inputs generated!");

    // Get Circuit Party
    let mut party = Circuits::new(party_id, INPUTS_PER_GPU_SIZE, url, Some(3001));
    let devices = party.get_devices();

    // Import to GPU
    let code_gpu = to_gpu(&input_bits_a, &input_bits_b, &devices);
    let mut res = alloc_res(INPUTS_PER_GPU_SIZE, &devices);
    println!("Data is on GPUs!");
    println!("Starting tests...");

    for _ in 0..10 {
        let code_gpu = code_gpu.clone();

        let now = Instant::now();
        party.bit_inject_neg_ot(&code_gpu, &mut res);
        party.synchronize_all();
        println!("compute time: {:?}", now.elapsed());

        let now = Instant::now();
        let result = open(&mut party, &mut res);
        println!("Open and transfer to CPU time: {:?}", now.elapsed());

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

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
