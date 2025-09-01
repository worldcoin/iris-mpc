use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use iris_mpc_common::{iris_db::iris::IrisCode, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use iris_mpc_cpu::protocol::{
    ops::galois_ring_pairwise_distance, shared_iris::GaloisRingSharedIris,
};
use itertools::Itertools;
use rand::seq::index::sample;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rayon::{prelude::*, ThreadPoolBuilder};

pub fn bench_galois_ring_pairwise_distance(c: &mut Criterion) {
    // Generate a dataset larger than CPU caches.
    let ram_size = 1_000_000_000; // 1 GB

    // --- Single-threaded Version ---

    let batch_size = 100;
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
        .map(Arc::new)
        .collect_vec();

    g.bench_function("Compute-bound", |b| {
        b.iter_batched(
            || {
                // Generate *one* batch of *cacheable* iris pairs.
                (0..batch_size)
                    .map(|_| Some((shares[0].clone(), shares[1].clone())))
                    .collect_vec()
            },
            |pairs| black_box(galois_ring_pairwise_distance(black_box(pairs))),
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
                        Some((shares[a].clone(), shares[b].clone()))
                    })
                    .collect_vec()
            },
            |pairs| black_box(galois_ring_pairwise_distance(black_box(pairs))),
            BatchSize::SmallInput,
        )
    });
    g.finish();

    // --- Parallel Version ---

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
                                .map(|_| Some((shares[0].clone(), shares[1].clone())))
                                .collect_vec()
                        })
                        .collect_vec()
                },
                |batches| {
                    batches
                        .into_par_iter()
                        .map(|pairs| black_box(galois_ring_pairwise_distance(black_box(pairs))))
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
                                    Some((shares[a].clone(), shares[b].clone()))
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                },
                |batches| {
                    batches
                        .into_par_iter()
                        .map(|pairs| black_box(galois_ring_pairwise_distance(black_box(pairs))))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        })
    });

    g.bench_function("RAM-bound with identical workloads per thread", |b| {
        pool.install(|| {
            b.iter_batched(
                || {
                    // Generate *multiple* batches of *non-cacheable* iris pairs.
                    (0..num_threads)
                        .map(|_| {
                            let mut rng = StdRng::seed_from_u64(42);
                            (0..batch_size)
                                .map(|_| {
                                    let a = rng.gen::<usize>() % shares.len();
                                    let b = rng.gen::<usize>() % shares.len();
                                    Some((shares[a].clone(), shares[b].clone()))
                                })
                                .collect_vec()
                        })
                        .collect_vec()
                },
                |batches| {
                    batches
                        .into_par_iter()
                        .map(|pairs| black_box(galois_ring_pairwise_distance(black_box(pairs))))
                        .collect::<Vec<_>>()
                },
                BatchSize::SmallInput,
            )
        })
    });

    g.finish();
}

fn generate_random_shared_irises(count: usize) -> Vec<Arc<GaloisRingSharedIris>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let iris = IrisCode::random_rng(&mut rng);
            // We only need one of the three shares for the benchmark structure.
            Arc::new(GaloisRingSharedIris::generate_shares_locally(&mut rng, iris)[0].clone())
        })
        .collect()
}

// Simulate dot-products as they occur in search_layer:
// DEPTH times we open a new candidate node, each contributing NEW_NODES_PER_LEVEL new nodes
// which are batched to compute distance to the source
// Constants are chosen according to staging runs as of late August 2025.

pub fn search_layer_like_calls(c: &mut Criterion) {
    let mut g = c.benchmark_group("search_layer_like_calls");
    g.sample_size(10usize);

    const INITIAL_RHS_COUNT: usize = 512;
    const DEPTH: usize = 356;
    const NEW_NODES_PER_LEVEL: usize = 65;
    let total_elements = INITIAL_RHS_COUNT + (DEPTH * NEW_NODES_PER_LEVEL);
    g.throughput(Throughput::Elements(total_elements as u64));

    g.bench_function("single-threaded", |b| {
        let source_iris = generate_random_shared_irises(1).pop().unwrap();

        let initial_rhs_irises = generate_random_shared_irises(INITIAL_RHS_COUNT);
        let all_irises = generate_random_shared_irises(total_elements);

        b.iter_batched(
            || {
                let mut rng = StdRng::seed_from_u64(42);
                let num_indices = DEPTH * NEW_NODES_PER_LEVEL;
                sample(&mut rng, all_irises.len(), num_indices).into_vec()
            },
            |mut indices| {
                let initial_pairs: Vec<_> = initial_rhs_irises
                    .iter()
                    .map(|rhs| Some((source_iris.clone(), rhs.clone())))
                    .collect();
                black_box(galois_ring_pairwise_distance(black_box(initial_pairs)));

                for _ in 0..DEPTH {
                    let curr_indices = indices.drain(0..NEW_NODES_PER_LEVEL).collect_vec();
                    let pairs_on_level: Vec<_> = (0..NEW_NODES_PER_LEVEL)
                        .map(|i| Some((source_iris.clone(), all_irises[curr_indices[i]].clone())))
                        .collect();

                    black_box(galois_ring_pairwise_distance(black_box(pairs_on_level)));
                }
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();

    let num_threads = num_cpus::get_physical();
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let mut g_parallel = c.benchmark_group(format!(
        "parallel search_layer-like calls -- num_threads={}",
        num_threads
    ));
    g_parallel.sample_size(10);
    g_parallel.throughput(Throughput::Elements((total_elements * num_threads) as u64));

    g_parallel.bench_function("identical order per thread", |b| {
        pool.install(|| {
            let thread_setups: Vec<_> = (0..num_threads)
                .map(|_| {
                    let source_iris = generate_random_shared_irises(1).pop().unwrap();
                    let initial_rhs_irises = generate_random_shared_irises(INITIAL_RHS_COUNT);
                    (source_iris, initial_rhs_irises)
                })
                .collect();
            let all_irises = generate_random_shared_irises(DEPTH * NEW_NODES_PER_LEVEL);

            b.iter_batched(
                || {
                    (0..num_threads)
                        .map(|_| {
                            let mut rng = StdRng::seed_from_u64(42);
                            let num_indices = DEPTH * NEW_NODES_PER_LEVEL;
                            sample(&mut rng, all_irises.len(), num_indices).into_vec()
                        })
                        .collect::<Vec<_>>()
                },
                |all_threads_indices| {
                    all_threads_indices
                        .par_iter()
                        .zip(thread_setups.par_iter())
                        .for_each(|(indices_for_thread, (source_iris, initial_rhs_irises))| {
                            let mut indices = indices_for_thread.clone();

                            let initial_pairs: Vec<_> = initial_rhs_irises
                                .iter()
                                .map(|rhs| Some((source_iris.clone(), rhs.clone())))
                                .collect();
                            black_box(galois_ring_pairwise_distance(black_box(initial_pairs)));

                            for _ in 0..DEPTH {
                                let curr_indices =
                                    indices.drain(0..NEW_NODES_PER_LEVEL).collect_vec();
                                let pairs_on_level: Vec<_> = curr_indices
                                    .iter()
                                    .map(|&index| {
                                        Some((source_iris.clone(), all_irises[index].clone()))
                                    })
                                    .collect();
                                black_box(galois_ring_pairwise_distance(black_box(pairs_on_level)));
                            }
                        });
                },
                BatchSize::SmallInput,
            )
        });
    });

    g_parallel.bench_function("different order per thread", |b| {
        pool.install(|| {
            let thread_setups: Vec<_> = (0..num_threads)
                .map(|_| {
                    let source_iris = generate_random_shared_irises(1).pop().unwrap();
                    let initial_rhs_irises = generate_random_shared_irises(INITIAL_RHS_COUNT);
                    (source_iris, initial_rhs_irises)
                })
                .collect();
            let all_irises = generate_random_shared_irises(DEPTH * NEW_NODES_PER_LEVEL);

            b.iter_batched(
                || {
                    // Setup closure: Generate the random indices for EACH thread.
                    (0..num_threads)
                        .map(|i| {
                            let mut rng = StdRng::seed_from_u64(i as u64);
                            let num_indices = DEPTH * NEW_NODES_PER_LEVEL;
                            sample(&mut rng, all_irises.len(), num_indices).into_vec()
                        })
                        .collect::<Vec<_>>()
                },
                |all_threads_indices| {
                    all_threads_indices
                        .par_iter()
                        .zip(thread_setups.par_iter())
                        .for_each(|(indices_for_thread, (source_iris, initial_rhs_irises))| {
                            let mut indices = indices_for_thread.clone();

                            let initial_pairs: Vec<_> = initial_rhs_irises
                                .iter()
                                .map(|rhs| Some((source_iris.clone(), rhs.clone())))
                                .collect();
                            black_box(galois_ring_pairwise_distance(black_box(initial_pairs)));

                            for _ in 0..DEPTH {
                                let curr_indices =
                                    indices.drain(0..NEW_NODES_PER_LEVEL).collect_vec();
                                let pairs_on_level: Vec<_> = curr_indices
                                    .iter()
                                    .map(|&index| {
                                        Some((source_iris.clone(), all_irises[index].clone()))
                                    })
                                    .collect();
                                black_box(galois_ring_pairwise_distance(black_box(pairs_on_level)));
                            }
                        });
                },
                BatchSize::SmallInput,
            )
        });
    });
    g_parallel.finish();
}

criterion_group!(
    benches,
    bench_galois_ring_pairwise_distance,
    search_layer_like_calls
);
criterion_main!(benches);
