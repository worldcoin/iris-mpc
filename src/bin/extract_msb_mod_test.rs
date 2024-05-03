use cudarc::driver::CudaDevice;
use gpu_iris_mpc::{
    setup::{
        iris_db::iris::IrisCodeArray,
        shamir::{Shamir, P},
    },
    threshold::protocol::{ChunkShare, Circuits},
};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

const INPUTS_PER_GPU_SIZE: usize = 125_000;
const B_BITS: u64 = 20;
const P2K: u64 = (P as u64) << B_BITS;

fn sample_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
    (0..size)
        .map(|_| {
            let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
            let neg = rng.gen::<bool>();
            if neg {
                x = P - x;
            }
            x
        })
        .collect::<Vec<_>>()
}

fn rep_share_fp<R: Rng>(value: u16, id: usize, rng: &mut R) -> (u16, u16) {
    let a = Shamir::random_fp(rng);
    let b = Shamir::random_fp(rng);
    let c = value as u32 + P as u32 + P as u32 - a as u32 - b as u32;
    let c = (c % P as u32) as u16;

    match id {
        0 => (a, c),
        1 => (b, a),
        2 => (c, b),
        _ => unreachable!(),
    }
}

fn rep_share_vec_fp<R: Rng>(value: &[u16], id: usize, rng: &mut R) -> (Vec<u16>, Vec<u16>) {
    let mut a = Vec::with_capacity(value.len());
    let mut b = Vec::with_capacity(value.len());
    for v in value.iter() {
        let (a_, b_) = rep_share_fp(*v, id, rng);
        a.push(a_);
        b.push(b_);
    }
    (a, b)
}

fn to_gpu(a: &[u16], b: &[u16], devices: &[Arc<CudaDevice>]) -> Vec<ChunkShare<u16>> {
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

fn real_result_msb(input: Vec<u16>) -> Vec<bool> {
    let mut res = Vec::with_capacity(input.len());
    for inp in input {
        let r = P2K - ((inp as u64) << B_BITS);
        let msb = r >> (B_BITS + 16) & 1 == 1;
        res.push(msb)
    }
    res
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // TODO
    let mut rng = StdRng::seed_from_u64(42);

    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);
    let n_devices = CudaDevice::count().unwrap() as usize;

    // Get inputs
    let code_dots = sample_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
    let (code_share_a, code_share_b) = rep_share_vec_fp(&code_dots, party_id, &mut rng);
    let real_result = real_result_msb(code_dots);
    println!("Random shared inputs generated!");

    // Get Circuit Party
    let party = Circuits::new(party_id, INPUTS_PER_GPU_SIZE, url, Some(3000));
    let devices = party.get_devices();

    // Import to GPU
    let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices);

    // Simulate Masks to be zero for this test
    let mask_gpu = party.allocate_buffer::<u64>(INPUTS_PER_GPU_SIZE);
    println!("Data is on GPUs!");
    println!("Starting tests...");

    for _ in 0..10 {
        let now = Instant::now();
        // TODO calculate and open here
        println!("Total time: {:?}", now.elapsed());
        // TODO compare to real result
    }

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
