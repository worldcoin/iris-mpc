use cudarc::driver::CudaDevice;
use gpu_iris_mpc::{
    setup::{
        iris_db::iris::{IrisCodeArray, MATCH_THRESHOLD_RATIO},
        shamir::{Shamir, P},
    },
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

//ceil(930 * 125_000 / 2048) * 2048
const INPUTS_PER_GPU_SIZE: usize = 116_250_624;
// const INPUTS_PER_GPU_SIZE: usize = 12_505_600;
const B_BITS: u64 = 20;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;
const P2K: u64 = (P as u64) << B_BITS;

fn sample_code_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
    (0..size)
        .map(|_| {
            let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
            let neg = rng.gen::<bool>();
            if neg {
                x = (P - x) % P;
            }
            x
        })
        .collect::<Vec<_>>()
}

fn sample_mask_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
    (0..size)
        .map(|_| rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16))
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

fn real_result_msb_reduce(code_input: Vec<u16>, mask_input: Vec<u16>) -> bool {
    assert_eq!(code_input.len(), mask_input.len());
    let mut res = false;
    for (c, m) in code_input.into_iter().zip(mask_input) {
        let r = ((m as u64) * A + P2K - ((c as u64) << B_BITS)) % P2K;
        let msb = r >> (B_BITS + 16 - 1) & 1 == 1;
        res |= msb;
    }
    res
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
    let code_dots = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
    let mask_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

    let (code_share_a, code_share_b) = rep_share_vec_fp(&code_dots, party_id, &mut rng);
    let (mask_share_a, mask_share_b) = rep_share_vec_fp(&mask_dots, party_id, &mut rng);
    let real_result = real_result_msb_reduce(code_dots, mask_dots);
    println!("Random shared inputs generated!");

    // Get Circuit Party
    let mut party = Circuits::new(party_id, INPUTS_PER_GPU_SIZE, url, Some(3001));
    let devices = party.get_devices();

    // Import to GPU
    let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices);
    let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices);
    println!("Data is on GPUs!");
    println!("Starting tests...");

    for _ in 0..10 {
        let code_gpu = code_gpu.clone();
        let mask_gpu = mask_gpu.clone();

        let now = Instant::now();
        party.compare_threshold_masked_many_with_or_tree(code_gpu, mask_gpu);
        party.synchronize_all();
        println!("compute time: {:?}", now.elapsed());

        let mut res = party.take_result_buffer();
        let now = Instant::now();
        let result = open(&mut party, &mut res[0]);
        party.return_result_buffer(res);
        println!("Open and transfer to CPU time: {:?}", now.elapsed());

        if result == real_result {
            println!("Test passed!");
        } else {
            println!("Test failed!");
        }
    }

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
