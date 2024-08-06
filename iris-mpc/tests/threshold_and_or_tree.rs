use cudarc::driver::{CudaDevice, CudaStream};
use iris_mpc::{
    helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
    setup::iris_db::iris::{IrisCodeArray, MATCH_THRESHOLD_RATIO},
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use static_assertions::const_assert;
use std::{env, sync::Arc};
use tokio::time::Instant;

// ceil(930 * 125_000 / 2048) * 2048
const INPUTS_PER_GPU_SIZE: usize = 116_250_624;
// const INPUTS_PER_GPU_SIZE: usize = 12_507_136;
const B_BITS: u64 = 16;
pub(crate) const B: u64 = 1 << B_BITS;
pub(crate) const A: u64 = ((1. - 2. * MATCH_THRESHOLD_RATIO) * B as f64) as u64;

fn sample_code_dots<R: Rng>(size: usize, rng: &mut R) -> Vec<u16> {
    (0..size)
        .map(|_| {
            let mut x = rng.gen_range::<u16, _>(0..=IrisCodeArray::IRIS_CODE_SIZE as u16);
            let neg = rng.gen::<bool>();
            if neg {
                x = u16::MAX - x + 1;
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

fn rep_share<R: Rng>(value: u16, id: usize, rng: &mut R) -> (u16, u16) {
    let a = rng.gen();
    let b = rng.gen();
    let c = value - a - b;

    match id {
        0 => (a, c),
        1 => (b, a),
        2 => (c, b),
        _ => unreachable!(),
    }
}

fn rep_share_vec<R: Rng>(value: &[u16], id: usize, rng: &mut R) -> (Vec<u16>, Vec<u16>) {
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

fn real_result_msb_reduce(code_input: Vec<u16>, mask_input: Vec<u16>) -> bool {
    assert_eq!(code_input.len(), mask_input.len());
    let mod_ = 1u64 << (16 + B_BITS);
    let mut res = false;
    for (c, m) in code_input.into_iter().zip(mask_input) {
        let r = ((m as u64) * A - ((c as u64) << B_BITS)) % mod_;
        let msb = r >> (B_BITS + 16 - 1) & 1 == 1;
        res |= msb;
    }
    res
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn test_threshold_and_or_tree() -> eyre::Result<()> {
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
    let code_dots = sample_code_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);
    let mask_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

    let (code_share_a, code_share_b) = rep_share_vec(&code_dots, party_id, &mut rng);
    let (mask_share_a, mask_share_b) = rep_share_vec(&mask_dots, party_id, &mut rng);
    let real_result = real_result_msb_reduce(code_dots, mask_dots);
    println!("Random shared inputs generated!");

    // Get Circuit Party
    let device_manager = Arc::new(DeviceManager::init());
    let ids = device_manager.get_ids_from_magic(0);
    let comms = device_manager.instantiate_network_from_ids(party_id, ids);
    let mut party = Circuits::new(
        party_id,
        INPUTS_PER_GPU_SIZE,
        INPUTS_PER_GPU_SIZE / 64,
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
    let code_gpu = to_gpu(&code_share_a, &code_share_b, &devices, &streams);
    let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices, &streams);
    println!("Data is on GPUs!");
    println!("Starting tests...");

    for _ in 0..10 {
        let code_gpu = code_gpu.clone();
        let mask_gpu = mask_gpu.clone();

        let now = Instant::now();
        party.compare_threshold_masked_many_with_or_tree(&code_gpu, &mask_gpu, &streams);
        println!("compute time: {:?}", now.elapsed());

        let mut res = party.take_result_buffer();
        let now = Instant::now();
        let result = open(&mut party, &mut res[0], &streams);
        party.synchronize_streams(&streams);
        party.return_result_buffer(res);
        println!("Open and transfer to CPU time: {:?}", now.elapsed());

        if result == real_result {
            println!("Test passed!");
        } else {
            println!("Test failed!");
        }
    }

    Ok(())
}
