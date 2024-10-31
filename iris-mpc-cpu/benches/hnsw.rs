use aes_prng::AesRng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use iris_mpc_cpu::{
    database_generators::{create_random_sharing, generate_galois_iris_shares},
    execution::local::LocalRuntime,
    hawkers::{
        galois_store::gr_create_ready_made_hawk_searcher,
        plaintext_store::PlaintextStore,
        session_based::{
            session_based_insert, session_based_ready_made_hawk_searcher, SessionBasedStorage,
        },
    },
    protocol::ops::{cross_compare, galois_ring_pairwise_distance, galois_ring_to_rep3},
};
use rand::SeedableRng;
use std::collections::HashMap;
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
            let searcher = HawkSearcher::default();

            for _ in 0..database_size {
                let raw_query = IrisCode::random_rng(&mut rng);
                let query = vector.prepare_query(raw_query.clone());
                let neighbors = searcher
                    .search_to_insert(&mut vector, &mut graph, &query)
                    .await;
                let inserted = vector.insert(&query).await;
                searcher
                    .insert_from_search_results(
                        &mut vector,
                        &mut graph,
                        &mut rng,
                        inserted,
                        neighbors,
                    )
                    .await;
            }
            (vector, graph)
        });

        group.bench_function(BenchmarkId::new("insert", database_size), |b| {
            b.to_async(&rt).iter_batched(
                || (vector.clone(), graph.clone()),
                |(mut db_vectors, mut graph)| async move {
                    let searcher = HawkSearcher::default();
                    let mut rng = AesRng::seed_from_u64(0_u64);
                    let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                    let query = db_vectors.prepare_query(on_the_fly_query);
                    let neighbors = searcher
                        .search_to_insert(&mut db_vectors, &mut graph, &query)
                        .await;
                    searcher
                        .insert_from_search_results(
                            &mut db_vectors,
                            &mut graph,
                            &mut rng,
                            query,
                            neighbors,
                        )
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

            let runtime = LocalRuntime::replicated_test_config();
            let ready_sessions = runtime.create_player_sessions().await.unwrap();

            let mut jobs = JoinSet::new();
            for (index, player) in runtime.identities.iter().enumerate() {
                let d1i = d1[index].clone();
                let d2i = d2[index].clone();
                let t1i = t1[index].clone();
                let t2i = t2[index].clone();
                let mut player_session = ready_sessions.get(player).unwrap().clone();
                jobs.spawn(async move {
                    cross_compare(&mut player_session, d1i, t1i, d2i, t2i)
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
            let runtime = LocalRuntime::replicated_test_config();
            let ready_sessions = runtime.create_player_sessions().await.unwrap();
            let mut rng = AesRng::seed_from_u64(0);
            let iris_db = IrisDB::new_random_rng(4, &mut rng).db;

            let x1 = generate_galois_iris_shares(&mut rng, iris_db[0].clone());
            let y1 = generate_galois_iris_shares(&mut rng, iris_db[2].clone());
            let x2 = generate_galois_iris_shares(&mut rng, iris_db[1].clone());
            let y2 = generate_galois_iris_shares(&mut rng, iris_db[3].clone());

            let mut jobs = JoinSet::new();
            for (index, player) in runtime.identities.iter().enumerate() {
                let x1 = x1[index].clone();
                let mut y1 = y1[index].clone();

                let x2 = x2[index].clone();
                let mut y2 = y2[index].clone();

                let mut player_session = ready_sessions.get(player).unwrap().clone();
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

fn bench_gr_ready_made_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("gr_ready_made_hnsw");
    group.sample_size(10);

    for database_size in [1, 10, 100, 1000, 10000, 100000] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let (_, secret_searcher) = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            gr_create_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap()
        });

        group.bench_function(
            BenchmarkId::new("gr-big-hnsw-insertions", database_size),
            |b| {
                b.to_async(&rt).iter_batched(
                    || secret_searcher.clone(),
                    |(mut db_vectors, mut db_graph)| async move {
                        let searcher = HawkSearcher::default();
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let query = db_vectors.prepare_query(raw_query);
                        let neighbors = searcher
                            .search_to_insert(&mut db_vectors, &mut db_graph, &query)
                            .await;
                        searcher
                            .insert_from_search_results(
                                &mut db_vectors,
                                &mut db_graph,
                                &mut rng,
                                query,
                                neighbors,
                            )
                            .await;
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
                    |(mut db_vectors, mut db_graph)| async move {
                        let searcher = HawkSearcher::default();
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let query = db_vectors.prepare_query(raw_query);
                        let neighbors = searcher
                            .search_to_insert(&mut db_vectors, &mut db_graph, &query)
                            .await;
                        searcher.is_match(&mut db_vectors, &neighbors).await;
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn bench_session_based_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_based_hnsw");
    group.sample_size(10);

    for database_size in [1, 10, 100, 1000, 10000, 100000] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        println!("Creating plaintext graph...");
        let (_, secret_data) = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            session_based_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap()
        });

        let ((p0_store, p1_store, p2_store), graph) = secret_data;
        let runtime = LocalRuntime::replicated_test_config();
        let identities = runtime.get_identities();
        let ready_sessions =
            rt.block_on(async move { runtime.create_player_sessions().await.unwrap() });

        let mut stores = HashMap::new();
        stores.insert(identities[0].clone(), p0_store);
        stores.insert(identities[1].clone(), p1_store);
        stores.insert(identities[2].clone(), p2_store);

        let mut rng = AesRng::seed_from_u64(0_u64);
        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);
        let mut shares = HashMap::new();
        shares.insert(identities[0].clone(), raw_query[0].clone());
        shares.insert(identities[1].clone(), raw_query[1].clone());
        shares.insert(identities[2].clone(), raw_query[2].clone());

        println!("Running the benchmark...");
        group.bench_function(BenchmarkId::new("insert", database_size), |b| {
            b.to_async(&rt).iter_batched(
                || {
                    (
                        ready_sessions.clone(),
                        shares.clone(),
                        stores.clone(),
                        graph.clone(),
                    )
                },
                |(sessions, shares, stores, graph)| {
                    let identities = identities.clone();
                    async move {
                        let identities = identities.clone();
                        let mut set = JoinSet::new();
                        for player_identity in identities.iter() {
                            let session = sessions.get(player_identity).unwrap().clone();
                            let store = stores.get(player_identity).unwrap().clone();
                            let raw_query = shares.get(player_identity).unwrap().clone();

                            let mut session_store = SessionBasedStorage::new(session, store);
                            let mut session_graph = graph.clone();
                            let query = session_store.prepare_query(raw_query);

                            set.spawn(async move {
                                let searcher = HawkSearcher::default();
                                let mut rng = AesRng::seed_from_u64(0_u64);
                                let _ = session_based_insert(
                                    &searcher,
                                    &mut session_store,
                                    &mut session_graph,
                                    &query,
                                    &mut rng,
                                )
                                .await;
                            });
                        }
                        let _ = set.join_all().await;
                    }
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group! {
    hnsw,
    bench_plaintext_hnsw,
    bench_gr_ready_made_hnsw,
    bench_hnsw_primitives,
    bench_gr_primitives,
    bench_session_based_hnsw,
}

criterion_main!(hnsw);
