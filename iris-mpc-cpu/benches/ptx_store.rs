//! The plaintext store is used for local tools and testing only. With this in mind, these benchmarks
//! confirm that using Rayon to open a batch of edges does in fact result in a speed up. There was a
//! concern that the Rayon scheduling overhead would exceed the computation time but benchmarks show
//! a 6-8x speedup over the serial version.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use iris_mpc_common::{iris_db::iris::IrisCode, VectorId};
use iris_mpc_cpu::hawkers::aby3::aby3_store::{DistanceOps, FhdOps};
use iris_mpc_cpu::hawkers::plaintext_store::SharedPlaintextStore;
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn generate_test_vectors(count: usize, seed: u64) -> Vec<(VectorId, Arc<IrisCode>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let iris = IrisCode::random_rng(&mut rng);
            (VectorId::new(i as u32, 0), Arc::new(iris))
        })
        .collect()
}

fn generate_query(seed: u64) -> IrisCode {
    let mut rng = StdRng::seed_from_u64(seed);
    IrisCode::random_rng(&mut rng)
}

/// Build a store pre-populated with the given vectors at their VectorIds.
async fn build_store(vectors: &[(VectorId, Arc<IrisCode>)]) -> SharedPlaintextStore {
    let store = SharedPlaintextStore::default();
    {
        let mut storage = store.storage.write().await;
        for (vid, code) in vectors {
            storage.insert(*vid, code.clone());
        }
    }
    store
}

/// Serial reimplementation of `SharedPlaintextStore::eval_distance_batch`.
///
/// Mirrors the library method exactly, minus the rayon parallelism: plain
/// sequential iteration over the vector lookups + distance evaluations.
async fn eval_distance_batch_serial(
    store: &SharedPlaintextStore,
    query: &Arc<IrisCode>,
    vectors: &[VectorId],
) -> eyre::Result<Vec<(u16, u16)>> {
    let storage = store.storage.read().await;
    let mode = store.distance_mode;
    vectors
        .iter()
        .map(|v| {
            let serial_id = v.serial_id();
            let code = storage.get_vector(v).ok_or_else(|| {
                eyre::eyre!("Vector ID not found in store for serial {}", serial_id)
            })?;
            Ok(FhdOps::plaintext_distance(code, query, mode))
        })
        .collect()
}

/// Rayon reimplementation of `SharedPlaintextStore::eval_distance_batch`.
///
/// Copies the current library implementation verbatim so both variants can be
/// benchmarked side by side without touching library code.
async fn eval_distance_batch_rayon(
    store: &SharedPlaintextStore,
    query: &Arc<IrisCode>,
    vectors: &[VectorId],
) -> eyre::Result<Vec<(u16, u16)>> {
    let storage = store.storage.read().await;
    let mode = store.distance_mode;
    vectors
        .par_iter()
        .with_min_len((vectors.len() / (rayon::current_num_threads() * 4)).max(1))
        .map(|v| {
            let serial_id = v.serial_id();
            let code = storage.get_vector(v).ok_or_else(|| {
                eyre::eyre!("Vector ID not found in store for serial {}", serial_id)
            })?;
            Ok(FhdOps::plaintext_distance(code, query, mode))
        })
        .collect()
}

/// Benchmark eval_distance_batch across neighborhood sizes, comparing the
/// serial implementation against the rayon-parallelized one (c74a230).
fn bench_eval_distance_batch_neighborhood_size(c: &mut Criterion) {
    let runtime = Runtime::new().expect("Failed to create tokio runtime");

    // Test with different batch sizes representing neighborhoods
    let batch_sizes = vec![10, 50, 100, 320];

    let mut group = c.benchmark_group("eval_distance_batch");

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size as u64));

        let vectors = generate_test_vectors(batch_size, 0xDEAD_BEEF);
        let query = Arc::new(generate_query(0xCAFE_BABE));
        let vector_refs: Vec<VectorId> = vectors.iter().map(|(vid, _)| *vid).collect();

        // Build the store once; the benched calls are read-only so it can be
        // shared across iterations (a cheap Arc clone per iteration).
        let store = runtime.block_on(build_store(&vectors));

        // Serial variant.
        group.bench_with_input(
            BenchmarkId::new("serial", format!("k={}", batch_size)),
            &batch_size,
            |b, &_| {
                b.to_async(&runtime).iter(|| {
                    let store = store.clone();
                    let query = query.clone();
                    let vector_refs = vector_refs.clone();
                    async move {
                        let distances = eval_distance_batch_serial(
                            &store,
                            black_box(&query),
                            black_box(&vector_refs),
                        )
                        .await
                        .expect("eval_distance_batch_serial failed");
                        black_box(distances);
                    }
                });
            },
        );

        // Rayon variant.
        group.bench_with_input(
            BenchmarkId::new("rayon", format!("k={}", batch_size)),
            &batch_size,
            |b, &_| {
                b.to_async(&runtime).iter(|| {
                    let store = store.clone();
                    let query = query.clone();
                    let vector_refs = vector_refs.clone();
                    async move {
                        let distances = eval_distance_batch_rayon(
                            &store,
                            black_box(&query),
                            black_box(&vector_refs),
                        )
                        .await
                        .expect("eval_distance_batch_rayon failed");
                        black_box(distances);
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_eval_distance_batch_neighborhood_size,);
criterion_main!(benches);
