//! Benchmarks for rotation_aware_pairwise_distance: original vs prerotated
//!
//! - original: 31 separate rotation calls per target
//! - prerotated: Precomputes query rotations + simple aligned dot products
//!
//! Run with: RUSTFLAGS='-C target-cpu=native' cargo bench -p iris-mpc-cpu --bench rotation_aware

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_common::ROTATIONS;
use iris_mpc_cpu::protocol::{
    ops::{rotation_aware_pairwise_distance, rotation_aware_pairwise_distance_prerotated},
    shared_iris::GaloisRingSharedIris,
};
use rand::thread_rng;

type ArcIris = Arc<GaloisRingSharedIris>;

fn generate_random_shared_irises(count: usize) -> Vec<ArcIris> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let iris = IrisCode::random_rng(&mut rng);
            Arc::new(GaloisRingSharedIris::generate_shares_locally(&mut rng, iris)[0].clone())
        })
        .collect()
}

/// Benchmark comparing original vs prerotated implementations
/// Both have the same interface: (query, targets) -> distances
pub fn bench_original_vs_prerotated(c: &mut Criterion) {
    let mut g = c.benchmark_group("rotation_aware");
    g.sample_size(50);

    let irises = generate_random_shared_irises(300);

    for target_count in [1, 16, 64, 128] {
        let elements_per_call = target_count as u64 * ROTATIONS as u64;
        g.throughput(Throughput::Elements(elements_per_call));

        let query = &irises[0];
        let targets: Vec<_> = (0..target_count).map(|i| &irises[i + 1]).collect();

        // Original implementation (31 separate rotation calls per target)
        g.bench_function(format!("original_targets_{}", target_count), |b| {
            b.iter(|| {
                black_box(rotation_aware_pairwise_distance(
                    query,
                    targets.iter().map(|t| Some(*t)),
                ))
            })
        });

        // Prerotated implementation (precomputes query rotations once)
        g.bench_function(format!("prerotated_targets_{}", target_count), |b| {
            b.iter(|| {
                black_box(rotation_aware_pairwise_distance_prerotated(
                    query,
                    targets.iter().map(|t| Some(*t)),
                ))
            })
        });
    }

    g.finish();
}

criterion_group!(benches, bench_original_vs_prerotated);
criterion_main!(benches);
