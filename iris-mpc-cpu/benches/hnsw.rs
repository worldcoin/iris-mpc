use aes_prng::AesRng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use iris_mpc_cpu::{
    database_generators::{create_random_sharing, generate_galois_iris_shares},
    execution::local::LocalRuntime,
    hawkers::{aby3_store::Aby3Store, plaintext_store::PlaintextStore},
    hnsw::{GraphMem, HnswSearcher},
    protocol::ops::{
        batch_signed_lift_vec, cross_compare, galois_ring_pairwise_distance, galois_ring_to_rep3,
    },
};
use rand::SeedableRng;
use tokio::task::JoinSet;

fn bench_plaintext_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("plaintext_hnsw");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for database_size in [100_usize, 1000, 10000] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let (vector, graph) = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            let mut vector = PlaintextStore::default();
            let mut graph = GraphMem::new();
            let searcher = HnswSearcher::default();

            for _ in 0..database_size {
                let raw_query = IrisCode::random_rng(&mut rng);
                let query = vector.prepare_query(raw_query.clone());
                searcher
                    .insert(&mut vector, &mut graph, &query, &mut rng)
                    .await;
            }
            (vector, graph)
        });

        group.bench_function(BenchmarkId::new("insert", database_size), |b| {
            b.to_async(&rt).iter_batched(
                || (vector.clone(), graph.clone()),
                |(mut db_vectors, mut graph)| async move {
                    let searcher = HnswSearcher::default();
                    let mut rng = AesRng::seed_from_u64(0_u64);
                    let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                    let query = db_vectors.prepare_query(on_the_fly_query);
                    searcher
                        .insert(&mut db_vectors, &mut graph, &query, &mut rng)
                        .await;
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_hnsw_primitives(c: &mut Criterion) {
    c.bench_function("lift-cross-mul", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        b.to_async(&rt).iter(|| async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            let d1 = create_random_sharing(&mut rng, 10_u16);
            let d2 = create_random_sharing(&mut rng, 10_u16);
            let t1 = create_random_sharing(&mut rng, 10_u16);
            let t2 = create_random_sharing(&mut rng, 10_u16);

            let runtime = LocalRuntime::mock_setup_with_grpc().await.unwrap();

            let mut jobs = JoinSet::new();
            for (index, player) in runtime.get_identities().iter().enumerate() {
                let d1i = d1[index].clone();
                let d2i = d2[index].clone();
                let t1i = t1[index].clone();
                let t2i = t2[index].clone();
                let mut player_session = runtime.sessions.get(player).unwrap().clone();
                jobs.spawn(async move {
                    let ds_and_ts = batch_signed_lift_vec(&mut player_session, vec![
                        d1i.clone(),
                        d2i.clone(),
                        t1i.clone(),
                        t2i.clone(),
                    ])
                    .await
                    .unwrap();
                    cross_compare(
                        &mut player_session,
                        ds_and_ts[0].clone(),
                        ds_and_ts[1].clone(),
                        ds_and_ts[2].clone(),
                        ds_and_ts[3].clone(),
                    )
                    .await
                    .unwrap()
                });
            }
            let _outputs = black_box(jobs.join_all().await);
        });
    });
}

fn bench_gr_primitives(c: &mut Criterion) {
    c.bench_function("gr-less-than", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        b.to_async(&rt).iter(|| async move {
            let runtime = LocalRuntime::mock_setup_with_grpc().await.unwrap();
            let mut rng = AesRng::seed_from_u64(0);
            let iris_db = IrisDB::new_random_rng(4, &mut rng).db;

            let x1 = generate_galois_iris_shares(&mut rng, iris_db[0].clone());
            let y1 = generate_galois_iris_shares(&mut rng, iris_db[2].clone());
            let x2 = generate_galois_iris_shares(&mut rng, iris_db[1].clone());
            let y2 = generate_galois_iris_shares(&mut rng, iris_db[3].clone());

            let mut jobs = JoinSet::new();
            for (index, player) in runtime.get_identities().iter().enumerate() {
                let x1 = x1[index].clone();
                let mut y1 = y1[index].clone();

                let x2 = x2[index].clone();
                let mut y2 = y2[index].clone();

                let mut player_session = runtime.sessions.get(player).unwrap().clone();
                jobs.spawn(async move {
                    y1.code.preprocess_iris_code_query_share();
                    y1.mask.preprocess_mask_code_query_share();
                    y2.code.preprocess_iris_code_query_share();
                    y2.mask.preprocess_mask_code_query_share();
                    let pairs = [(x1, y1), (x2, y2)];
                    let ds_and_ts = galois_ring_pairwise_distance(&mut player_session, &pairs)
                        .await
                        .unwrap();
                    let ds_and_ts = galois_ring_to_rep3(&mut player_session, ds_and_ts)
                        .await
                        .unwrap();
                    let ds_and_ts = batch_signed_lift_vec(&mut player_session, ds_and_ts)
                        .await
                        .unwrap();
                    cross_compare(
                        &mut player_session,
                        ds_and_ts[0].clone(),
                        ds_and_ts[1].clone(),
                        ds_and_ts[2].clone(),
                        ds_and_ts[3].clone(),
                    )
                    .await
                    .unwrap();
                });
            }
            let _outputs = black_box(jobs.join_all().await);
        });
    });
}

/// To run this benchmark, you need to generate the data first by running the
/// following commands:
///
/// cargo run --release --bin generate_benchmark_data
fn bench_gr_ready_made_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("gr_ready_made_hnsw");
    group.sample_size(10);

    for database_size in [1, 10, 100, 1000, 10_000, 100_000] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let secret_searcher = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            Aby3Store::lazy_setup_from_files_with_grpc(
                "./data/store.ndjson",
                &format!("./data/graph_{}.dat", database_size),
                &mut rng,
                database_size,
                false,
            )
            .await
        });

        if let Err(e) = secret_searcher {
            eprintln!("bench_gr_ready_made_hnsw failed. {e:?}");
            rt.shutdown_timeout(std::time::Duration::from_secs(5));
            return;
        }
        let (_, secret_searcher) = secret_searcher.unwrap();

        group.bench_function(
            BenchmarkId::new("gr-big-hnsw-insertions", database_size),
            |b| {
                b.to_async(&rt).iter_batched(
                    || secret_searcher.clone(),
                    |vectors_graphs| async move {
                        let searcher = HnswSearcher::default();
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let mut jobs = JoinSet::new();

                        for (vector_store, graph_store) in vectors_graphs.into_iter() {
                            let mut vector_store = vector_store;
                            let mut graph_store = graph_store;

                            let player_index = vector_store.get_owner_index();
                            let query = vector_store.prepare_query(raw_query[player_index].clone());
                            let searcher = searcher.clone();
                            let mut rng = rng.clone();
                            jobs.spawn(async move {
                                searcher
                                    .insert(&mut vector_store, &mut graph_store, &query, &mut rng)
                                    .await;
                            });
                        }
                        jobs.join_all().await;
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_function(
            BenchmarkId::new("gr-big-hnsw-searches", database_size),
            |b| {
                b.to_async(&rt).iter_batched(
                    || secret_searcher.clone(),
                    |vectors_graphs| async move {
                        let searcher = HnswSearcher::default();
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let mut jobs = JoinSet::new();
                        for (vector_store, graph_store) in vectors_graphs.into_iter() {
                            let mut vector_store = vector_store;
                            let mut graph_store = graph_store;
                            let player_index = vector_store.get_owner_index();
                            let query = vector_store.prepare_query(raw_query[player_index].clone());
                            let searcher = searcher.clone();
                            jobs.spawn(async move {
                                let neighbors = searcher
                                    .search(&mut vector_store, &mut graph_store, &query, 1)
                                    .await;
                                searcher.is_match(&mut vector_store, &[neighbors]).await;
                            });
                        }
                        jobs.join_all().await;
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

criterion_group! {
    hnsw,
    bench_plaintext_hnsw,
    bench_gr_ready_made_hnsw,
    bench_hnsw_primitives,
    bench_gr_primitives,
}

criterion_main!(hnsw);
