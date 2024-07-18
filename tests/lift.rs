use cudarc::driver::{CudaDevice, CudaStream};
use gpu_iris_mpc::{
    helpers::{
        device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync,
        task_monitor::TaskMonitor,
    },
    setup::iris_db::iris::IrisCodeArray,
    threshold_ring::protocol::{ChunkShare, ChunkShareView, Circuits},
};
use itertools::izip;
use rand::{rngs::StdRng, Rng, SeedableRng};
use static_assertions::const_assert;
use std::{env, sync::Arc};
use tokio::time::{self, Instant};

// ceil(930 * 125_000 / 2048) * 2048
// const INPUTS_PER_GPU_SIZE: usize = 116_250_624;
const INPUTS_PER_GPU_SIZE: usize = 12_507_136;

fn to_view<T>(inp: &[ChunkShare<T>]) -> Vec<ChunkShareView<T>> {
    let mut res = Vec::with_capacity(inp.len());
    for inp in inp {
        res.push(inp.as_view());
    }
    res
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

fn real_result_msb(mask_input: Vec<u16>) -> Vec<u32> {
    mask_input.into_iter().map(|x| (x as u32)).collect()
}

fn open(
    party: &mut Circuits,
    x: &mut [ChunkShareView<u32>],
    corrections: &mut [ChunkShareView<u16>],
    streams: &[CudaStream],
) -> Vec<u32> {
    let n_devices = x.len();
    let mut res_a = Vec::with_capacity(n_devices);
    let mut res_b = Vec::with_capacity(n_devices);
    let mut res_c = Vec::with_capacity(n_devices);
    let mut corr_a = Vec::with_capacity(n_devices);
    let mut corr_b = Vec::with_capacity(n_devices);
    let mut corr_c = Vec::with_capacity(n_devices);

    let devices = party.get_devices();
    for (idx, (res, corr)) in izip!(x.iter(), corrections.iter()).enumerate() {
        res_a.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
        res_b.push(dtoh_on_stream_sync(&res.b, &devices[idx], &streams[idx]).unwrap());
        corr_a.push(dtoh_on_stream_sync(&corr.a, &devices[idx], &streams[idx]).unwrap());
        corr_b.push(dtoh_on_stream_sync(&corr.b, &devices[idx], &streams[idx]).unwrap());
    }
    cudarc::nccl::result::group_start().unwrap();
    for (idx, (res, corr)) in izip!(x.iter(), corrections.iter()).enumerate() {
        party
            .send_view(&res.b, party.next_id(), idx, streams)
            .unwrap();
        party
            .send_view_u16(&corr.b, party.next_id(), idx, streams)
            .unwrap();
    }
    for (idx, (res, corr)) in izip!(x.iter_mut(), corrections.iter_mut()).enumerate() {
        party
            .receive_view(&mut res.a, party.prev_id(), idx, streams)
            .unwrap();
        party
            .receive_view_u16(&mut corr.a, party.prev_id(), idx, streams)
            .unwrap();
    }
    cudarc::nccl::result::group_end().unwrap();
    for (idx, (res, corr)) in izip!(x, corrections).enumerate() {
        res_c.push(dtoh_on_stream_sync(&res.a, &devices[idx], &streams[idx]).unwrap());
        corr_c.push(dtoh_on_stream_sync(&corr.a, &devices[idx], &streams[idx]).unwrap());
    }

    let mut result = Vec::with_capacity(n_devices * INPUTS_PER_GPU_SIZE);
    for (mut res_a, res_b, res_c, corr_a, corr_b, corr_c) in
        izip!(res_a, res_b, res_c, corr_a, corr_b, corr_c)
    {
        assert_eq!(res_a.len(), INPUTS_PER_GPU_SIZE);
        assert_eq!(res_b.len(), INPUTS_PER_GPU_SIZE);
        assert_eq!(res_c.len(), INPUTS_PER_GPU_SIZE);
        assert_eq!(corr_a.len(), INPUTS_PER_GPU_SIZE * 2);
        assert_eq!(corr_b.len(), INPUTS_PER_GPU_SIZE * 2);
        assert_eq!(corr_c.len(), INPUTS_PER_GPU_SIZE * 2);

        for (res_a, res_b, res_c, corr_a1, corr_b1, corr_c1, corr_a2, corr_b2, corr_c2) in izip!(
            &mut res_a,
            res_b,
            res_c,
            corr_a.iter().take(INPUTS_PER_GPU_SIZE),
            corr_b.iter().take(INPUTS_PER_GPU_SIZE),
            corr_c.iter().take(INPUTS_PER_GPU_SIZE),
            corr_a.iter().skip(INPUTS_PER_GPU_SIZE),
            corr_b.iter().skip(INPUTS_PER_GPU_SIZE),
            corr_c.iter().skip(INPUTS_PER_GPU_SIZE),
        ) {
            let corr1 = *corr_a1 + corr_b1 + corr_c1;
            let corr2 = *corr_a2 + corr_b2 + corr_c2;
            assert!(corr1 == 0 || corr1 == 1);
            assert!(corr2 == 0 || corr2 == 1);
            let mut res = *res_a + res_b + res_c;
            res -= (corr1 as u32) << 16;
            res -= (corr2 as u32) << 17;
            *res_a = res;
        }
        result.extend(res_a);
    }

    assert_eq!(result.len(), n_devices * INPUTS_PER_GPU_SIZE);
    result
}

#[tokio::test]
#[ignore]
async fn test_lift() -> eyre::Result<()> {
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
    let url = env::var("PEER_URL")?;
    let n_devices = CudaDevice::count()? as usize;

    // Get inputs
    let mask_dots = sample_mask_dots(INPUTS_PER_GPU_SIZE * n_devices, &mut rng);

    let (mask_share_a, mask_share_b) = rep_share_vec(&mask_dots, party_id, &mut rng);
    let real_result = real_result_msb(mask_dots);
    println!("Random shared inputs generated!");

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

    // Import to GPU
    let mask_gpu = to_gpu(&mask_share_a, &mask_share_b, &devices, &streams);
    println!("Data is on GPUs!");
    println!("Starting tests...");

    for _ in 0..10 {
        server_tasks.check_tasks();

        // Simulate Masks to be zero for this test
        let x_ = party.allocate_buffer::<u32>(INPUTS_PER_GPU_SIZE);
        let mut x = to_view(&x_);
        let correction_ = party.allocate_buffer::<u16>(INPUTS_PER_GPU_SIZE * 2);
        let mut correction = to_view(&correction_);
        let mask_gpu = mask_gpu.clone();

        let now = Instant::now();
        party.lift_mpc(&mask_gpu, &mut x, &mut correction, &streams);
        println!("compute time: {:?}", now.elapsed());

        let now = Instant::now();
        let result = open(&mut party, &mut x, &mut correction, &streams);
        party.synchronize_streams(&streams);
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

    server_tasks.abort_all();
    time::sleep(time::Duration::from_secs(5)).await;
    server_tasks.check_tasks_finished();
    Ok(())
}
