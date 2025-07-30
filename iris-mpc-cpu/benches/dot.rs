use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use iris_mpc_common::{iris_db::iris::IrisCode, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use iris_mpc_cpu::protocol::{
    ops::galois_ring_pairwise_distance, shared_iris::GaloisRingSharedIris,
};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rayon::{prelude::*, ThreadPoolBuilder};

pub fn bench_galois_ring_pairwise_distance(c: &mut Criterion) {
    // Generate a dataset larger than CPU caches.
    let ram_size = 1_000_000_000; // 1 GB

    // --- Single-threaded Version ---

    let batch_size = 1;
    let mut g = c.benchmark_group(format!(
        "galois_ring_pairwise_distance * batch_size={batch_size} * single-threaded"
    ));
    g.throughput(Throughput::Elements(batch_size));

    let iris_size = (IRIS_CODE_LENGTH + MASK_CODE_LENGTH) * size_of::<u16>();
    let dataset_size = ram_size / iris_size;
    let rng = &mut thread_rng();

    let shares = (0..dataset_size / 3)
        .flat_map(|_| {
            let iris = IrisCode::random_rng(rng);
            // Mash up the 3 party shares; ok for benchmarking.
            GaloisRingSharedIris::generate_shares_locally(rng, iris)
        })
        .collect_vec();

    g.bench_function("Compute-bound", |b| {
        b.iter_batched(
            || {
                // Generate *one* batch of *cacheable* iris pairs.
                (0..batch_size)
                    .map(|_| Some((&shares[0], &shares[1])))
                    .collect_vec()
            },
            |pairs| galois_ring_pairwise_distance(black_box(&pairs)),
            BatchSize::SmallInput,
        )
    });

    g.bench_function("RAM-bound", |b| {
        b.iter_batched(
            || {
                // Generate *one* batch of *non-cacheable* iris pairs.
                (0..batch_size)
                    .map(|_| {
                        let a = rng.gen::<usize>() % shares.len();
                        let b = rng.gen::<usize>() % shares.len();
                        Some((&shares[a], &shares[b]))
                    })
                    .collect_vec()
            },
            |pairs| galois_ring_pairwise_distance(black_box(&pairs)),
            BatchSize::SmallInput,
        )
    });
    g.finish();

    // --- Parallel Version ---

    let batch_size = 8;
    let num_threads = num_cpus::get_physical();

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let mut g = c.benchmark_group(format!(
        "galois_ring_pairwise_distance * batch_size={batch_size} * num_threads={num_threads}"
    ));
    g.throughput(Throughput::Elements(batch_size * num_threads as u64));

    g.bench_function("Compute-bound", |b| {
        pool.install(|| {
            b.iter_batched(
                || {
                    // Generate *multiple* batches of *cacheable* iris pairs.
                    (0..num_threads)
                        .map(|_| {
                            (0..batch_size)
                                .map(|_| Some((&shares[0], &shares[1])))
                                .collect_vec()
                        })
                        .collect_vec()
                },
                |batches| {
                    batches
                        .par_iter()
                        .map(|pairs| galois_ring_pairwise_distance(black_box(pairs)))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        })
    });

    g.bench_function("RAM-bound", |b| {
        pool.install(|| {
            b.iter_batched(
                || {
                    // Generate *multiple* batches of *non-cacheable* iris pairs.
                    let rng = &mut thread_rng();
                    (0..num_threads)
                        .map(|_| {
                            (0..batch_size)
                                .map(|_| {
                                    let a = rng.gen::<usize>() % shares.len();
                                    let b = rng.gen::<usize>() % shares.len();
                                    Some((&shares[a], &shares[b]))
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                },
                |batches| {
                    batches
                        .par_iter()
                        .map(|pairs| galois_ring_pairwise_distance(black_box(pairs)))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        })
    });

    g.finish();
}

criterion_group!(benches, bench_galois_ring_pairwise_distance);
criterion_main!(benches);
