use std::collections::HashMap;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use iris_mpc_common::galois_engine::degree4::{rotation_aware_trick_dot_padded, IrisRotation};
use iris_mpc_common::{
    iris_db::iris::IrisCode, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, PRE_PROC_IRIS_CODE_LENGTH,
};
use iris_mpc_common::{IrisVectorId, ROTATIONS};
use iris_mpc_cpu::execution::hawk_main::iris_worker::init_workers;
use iris_mpc_cpu::hawkers::shared_irises::SharedIrises;
use iris_mpc_cpu::protocol::{
    ops::galois_ring_pairwise_distance, shared_iris::GaloisRingSharedIris,
};
use itertools::Itertools;
use rand::seq::index::sample;
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
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

pub fn bench_trick_dot(c: &mut Criterion) {
    let batch_size = 100;
    let rng = &mut thread_rng();

    let mut g = c.benchmark_group("trick_dot_vs_rotation_aware_w_cache");
    g.sample_size(10);
    g.throughput(Throughput::Elements(batch_size));

    // Prepare a large dataset of random iris codes and their shares
    // should be divisible by 3
    let dataset_size = 99999;
    let iris_codes: Vec<_> = (0..dataset_size / 3)
        .flat_map(|_| {
            let iris = IrisCode::random_rng(rng);
            // Mash up the 3 party shares; ok for benchmarking.
            GaloisRingSharedIris::generate_shares_locally(rng, iris)
        })
        .collect();

    // Prepare random arrays for padded benchmarks
    let random_arrays: Vec<[u16; PRE_PROC_IRIS_CODE_LENGTH]> = (0..dataset_size)
        .map(|_| {
            let mut arr = [0u16; PRE_PROC_IRIS_CODE_LENGTH];
            for elem in arr.iter_mut() {
                *elem = rng.gen();
            }
            arr
        })
        .collect();

    // --- Compute-bound (cacheable) version ---
    g.bench_function("trick_dot_compute_bound", |b| {
        let left = &iris_codes[0];
        let right = &iris_codes[1];
        b.iter_batched(
            || (0..batch_size).map(|_| (left, right)).collect::<Vec<_>>(),
            |pairs| {
                for (l, r) in pairs {
                    black_box(l.code.trick_dot(&r.code));
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("rotation_aware_trick_dot_compute_bound", |b| {
        let left = &iris_codes[0];
        let right = &iris_codes[1];
        b.iter_batched(
            || (0..batch_size).map(|_| (left, right)).collect::<Vec<_>>(),
            |pairs| {
                for (l, r) in pairs {
                    black_box(
                        l.code
                            .rotation_aware_trick_dot(&r.code, &IrisRotation::Left(12)),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("rotation_aware_trick_dot_padded_compute_bound", |b| {
        let right = &iris_codes[0].code.coefs;
        let preprocessed_data = &random_arrays[1];
        b.iter_batched(
            || {
                (0..batch_size)
                    .map(|_| (preprocessed_data, right))
                    .collect::<Vec<_>>()
            },
            |pairs| {
                for (preprocessed_data, right) in pairs {
                    black_box(rotation_aware_trick_dot_padded(
                        preprocessed_data,
                        right,
                        &IrisRotation::Left(12),
                    ));
                }
            },
            BatchSize::SmallInput,
        )
    });

    // --- RAM-bound (non-cacheable) version ---
    g.bench_function("ram_bound", |b| {
        b.iter_batched(
            || {
                (0..batch_size)
                    .map(|_| {
                        let a = rng.gen_range(0..dataset_size);
                        let b = rng.gen_range(0..dataset_size);
                        (&iris_codes[a], &iris_codes[b])
                    })
                    .collect::<Vec<_>>()
            },
            |pairs| {
                for (l, r) in pairs {
                    black_box(l.code.trick_dot(&r.code));
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("rotation_aware_trick_dot_ram_bound", |b| {
        b.iter_batched(
            || {
                (0..batch_size)
                    .map(|_| {
                        let a = rng.gen_range(0..dataset_size);
                        let b = rng.gen_range(0..dataset_size);
                        let c = rng.gen_range(1..16);
                        (&iris_codes[a], &iris_codes[b], c)
                    })
                    .collect::<Vec<_>>()
            },
            |triples| {
                for (l, r, rot) in triples {
                    black_box(
                        l.code
                            .rotation_aware_trick_dot(&r.code, &IrisRotation::Left(rot)),
                    );
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.bench_function("rotation_aware_trick_dot_padded_ram_bound", |b| {
        b.iter_batched(
            || {
                (0..batch_size)
                    .map(|_| {
                        let a = rng.gen_range(0..dataset_size);
                        let b = rng.gen_range(0..dataset_size);
                        let c = rng.gen_range(1..16);
                        (&random_arrays[b], &iris_codes[a], c)
                    })
                    .collect::<Vec<_>>()
            },
            |triples| {
                for (preprocessed_data, r, rot) in triples {
                    black_box(rotation_aware_trick_dot_padded(
                        preprocessed_data,
                        &r.code.coefs,
                        &IrisRotation::Left(rot),
                    ));
                }
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

// operation: perform trick dot for: all rotations of code A vs all codes in set B
pub fn bench_batch_trick_dot(c: &mut Criterion) {
    let batch_size = 100;
    let rng = &mut thread_rng();

    let mut g = c.benchmark_group("batch_trick_dot");
    g.sample_size(50);
    g.throughput(Throughput::Elements(batch_size));

    // Prepare a large dataset of random iris codes and their shares
    // should be divisible by 3
    let dataset_size = 99999;
    let dist = Uniform::new(0, dataset_size);
    let iris_codes: Vec<_> = (0..dataset_size / 3)
        .flat_map(|_| {
            let iris = IrisCode::random_rng(rng);
            // Mash up the 3 party shares; ok for benchmarking.
            GaloisRingSharedIris::generate_shares_locally(rng, iris)
        })
        .collect();

    // Prepare random arrays for padded benchmarks
    let random_arrays: Vec<[u16; PRE_PROC_IRIS_CODE_LENGTH]> = (0..dataset_size)
        .map(|_| {
            let mut arr = [0u16; PRE_PROC_IRIS_CODE_LENGTH];
            for elem in arr.iter_mut() {
                *elem = rng.gen();
            }
            arr
        })
        .collect();

    // --- RAM-bound (non-cacheable) version ---

    const NEAREST_NEIGHBORS: [usize; 3] = [32, 64, 128];

    for &nearest_neighbors in &NEAREST_NEIGHBORS {
        g.bench_function(format!("regular_{}", nearest_neighbors), |b| {
            b.iter_batched(
                || {
                    (0..batch_size)
                        .map(|_| {
                            let a = dist.sample(rng);
                            let mut b = vec![];
                            for _ in 0..nearest_neighbors {
                                let idx = dist.sample(rng);
                                b.push(&iris_codes[idx]);
                            }
                            (&iris_codes[a], b)
                        })
                        .collect::<Vec<_>>()
                },
                |input| {
                    for (l, set) in input {
                        for v in set {
                            for rot in IrisRotation::all() {
                                black_box(l.code.rotation_aware_trick_dot(&v.code, &rot));
                            }
                        }
                    }
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_function(format!("padded_{}", nearest_neighbors), |b| {
            b.iter_batched(
                || {
                    (0..batch_size)
                        .map(|_| {
                            let a = dist.sample(rng);
                            let mut b = vec![];
                            for _ in 0..nearest_neighbors {
                                let idx = dist.sample(rng);
                                b.push(&iris_codes[idx]);
                            }
                            (&random_arrays[a], b)
                        })
                        .collect::<Vec<_>>()
                },
                |input| {
                    for (l, set) in input {
                        for v in set {
                            for rot in IrisRotation::all() {
                                black_box(rotation_aware_trick_dot_padded(l, &v.code.coefs, &rot));
                            }
                        }
                    }
                },
                BatchSize::SmallInput,
            )
        });
    }

    g.finish();
}

pub fn bench_worker_pool(c: &mut Criterion) {
    let rng = &mut thread_rng();
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut g = c.benchmark_group("batch_trick_dot_iris_worker");
    g.sample_size(10);

    // Prepare a large dataset of random iris codes and their shares
    // should be divisible by 3
    let dataset_size = 99999;
    let iris_codes: Vec<_> = (0..dataset_size / 3)
        .flat_map(|_| {
            let rng = &mut thread_rng();
            let iris = IrisCode::random_rng(rng);
            GaloisRingSharedIris::generate_shares_locally(rng, iris)
        })
        .map(Arc::new)
        .collect();

    let num_iris_codes = iris_codes.len();
    let dist = Uniform::new(0, num_iris_codes);

    let points_map: HashMap<IrisVectorId, Arc<GaloisRingSharedIris>> = HashMap::new();
    let shared_irises = SharedIrises::new(
        points_map,
        Arc::new(GaloisRingSharedIris::default_for_party(0)),
    )
    .to_arc();
    let mut pool = init_workers(0, shared_irises, true);

    // similar to numa_realloc
    for (idx, iris) in iris_codes.iter().enumerate() {
        pool.insert(IrisVectorId::from_0_index(idx as u32), iris.clone())
            .unwrap();
    }

    let _ = rt.block_on(pool.wait_completion());

    for nearest_neighbors in [4096] {
        g.throughput(Throughput::Elements(
            nearest_neighbors as u64 * ROTATIONS as u64,
        ));

        for chunk_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            if chunk_size > nearest_neighbors {
                continue;
            }

            g.bench_function(format!("nn_{nearest_neighbors}_cs_{chunk_size}",), |b| {
                b.iter_batched(
                    || {
                        let a = dist.sample(rng);
                        let mut b = vec![];
                        for _ in 0..nearest_neighbors {
                            let idx = dist.sample(rng);
                            b.push(IrisVectorId::from_0_index(idx as u32));
                        }
                        (iris_codes[a].clone(), b)
                    },
                    |(query, targets)| {
                        let _ = std::hint::black_box(
                            rt.block_on(pool.bench_batch_dot(chunk_size, query, targets)),
                        );
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_worker_pool,
    bench_batch_trick_dot,
    bench_trick_dot,
    bench_galois_ring_pairwise_distance,
    search_layer_like_calls
);
criterion_main!(benches);
