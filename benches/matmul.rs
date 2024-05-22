use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use gpu_iris_mpc::{device_manager::DeviceManager, device_ptrs, preprocess_query, setup::shamir::P, ShareDB};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn random_vec(n: usize, m: usize, max_value: u32) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    (0..n * m)
        .map(|_| rng.gen_range(0..max_value) as u16)
        .collect()
}

const RNG_SEED: u64 = 42;
const DB_SIZE: usize = 8 * 300_000;
const QUERY_SIZE: usize = 930;
const WIDTH: usize = 12800;

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");

    let db = random_vec(DB_SIZE, WIDTH, P as u32);
    let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
    let device_manager = DeviceManager::init();

    let mut engine = ShareDB::init(
        0,
        device_manager.clone(),
        1,
        DB_SIZE,
        QUERY_SIZE,
        ([0u32; 8], [0u32; 8]),
        None,
        None,
        None,
    );
    let preprocessed_query = preprocess_query(&query);
    let streams = device_manager.fork_streams();
    let blass = device_manager.create_cublas(&streams);
    let db_slices = engine.load_db(&db);

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    group.sample_size(10);
    
    group.bench_function(format!("matmul {} x {}", DB_SIZE, QUERY_SIZE), |b| {
        b.iter(|| {
            let preprocessed_query = device_manager.htod_transfer_query(&preprocessed_query, &streams);
            let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
            engine.dot(&preprocessed_query, &(device_ptrs(&db_slices.0 .0), device_ptrs(&db_slices.0 .1)), &streams, &blass);
            engine.dot_reduce(&query_sums, &(device_ptrs(&db_slices.1 .0), device_ptrs(&db_slices.1 .1)), &streams);
            device_manager.await_streams(&streams);
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
