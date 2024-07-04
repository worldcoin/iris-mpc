use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use gpu_iris_mpc::{
    dot::{
        device_manager::DeviceManager,
        share_db::{preprocess_query, ShareDB},
    },
    helpers::device_ptrs,
    setup::shamir::P,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

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
    let device_manager = Arc::new(DeviceManager::init());

    let mut engine = ShareDB::init(
        0,
        device_manager.clone(),
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
    let db_slices = engine.load_db(&db, DB_SIZE, DB_SIZE, false);
    let db_sizes = vec![DB_SIZE; 8];

    group.throughput(Throughput::Elements((DB_SIZE * QUERY_SIZE / 31) as u64));
    group.sample_size(10);

    group.bench_function(format!("matmul {} x {}", DB_SIZE, QUERY_SIZE), |b| {
        b.iter(|| {
            let preprocessed_query =
                device_manager.htod_transfer_query(&preprocessed_query, &streams);
            let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
            engine.dot(
                &preprocessed_query,
                &(device_ptrs(&db_slices.0 .0), device_ptrs(&db_slices.0 .1)),
                &db_sizes,
                &streams,
                &blass,
            );
            engine.dot_reduce(
                &query_sums,
                &(device_ptrs(&db_slices.1 .0), device_ptrs(&db_slices.1 .1)),
                &db_sizes,
                &streams,
            );
            device_manager.await_streams(&streams);
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
