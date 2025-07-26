use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use iris_mpc_common::{iris_db::iris::IrisCode, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use iris_mpc_cpu::protocol::{
    ops::galois_ring_pairwise_distance, shared_iris::GaloisRingSharedIris,
};
use itertools::Itertools;
use rand::{thread_rng, Rng};

pub fn bench_galois_ring_pairwise_distance(c: &mut Criterion) {
    // Generate a dataset larger than CPU caches.
    let ram_size = 1_000_000_000; // 1 GB
    let batch_size = 1;

    let mut c = c.benchmark_group("galois_ring_pairwise_distance");
    c.throughput(Throughput::Elements(batch_size));

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

    c.bench_function("Compute", |b| {
        let cached_pairs = vec![Some((&shares[0], &shares[1])); batch_size as usize];
        b.iter_with_large_drop(|| galois_ring_pairwise_distance(black_box(&cached_pairs)))
    });

    c.bench_function("RAM-bound", |b| {
        b.iter_batched(
            || {
                // Generate a batch of iris pairs.
                (0..batch_size)
                    .map(|_| {
                        // Randomly select two shares from the dataset.
                        let a = rng.gen::<usize>() % shares.len();
                        let b = rng.gen::<usize>() % shares.len();
                        Some((&shares[a], &shares[b]))
                    })
                    .collect_vec()
            },
            |uncached_pairs| galois_ring_pairwise_distance(black_box(&uncached_pairs)),
            BatchSize::SmallInput,
        )
    });

    c.finish();
}

criterion_group!(benches, bench_galois_ring_pairwise_distance);
criterion_main!(benches);
