use aes_prng::AesRng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use iris_mpc_cpu::{
    database_generators::{create_random_sharing, generate_galois_iris_shares},
    hawkers::{galois_store::gr_create_ready_made_hawk_searcher, plaintext_store::PlaintextStore},
    next_gen_protocol::ng_worker::{
        gr_replicated_pairwise_distance, gr_to_rep3, ng_cross_compare, LocalRuntime,
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

        let plain_searcher = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            let vector_store = PlaintextStore::default();
            let graph_store = GraphMem::new();
            let mut plain_searcher = HawkSearcher::new(vector_store, graph_store, &mut rng);

            for _ in 0..database_size {
                let raw_query = IrisCode::random_rng(&mut rng);
                let query = plain_searcher.vector_store.prepare_query(raw_query.clone());
                let neighbors = plain_searcher.search_to_insert(&query).await;
                let inserted = plain_searcher.vector_store.insert(&query).await;
                plain_searcher
                    .insert_from_search_results(inserted, neighbors)
                    .await;
            }
            plain_searcher
        });

        group.bench_function(BenchmarkId::new("insert", database_size), |b| {
            b.to_async(&rt).iter_batched(
                || plain_searcher.clone(),
                |mut my_db| async move {
                    let mut rng = AesRng::seed_from_u64(0_u64);
                    let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                    let query = my_db.vector_store.prepare_query(on_the_fly_query);
                    let neighbors = my_db.search_to_insert(&query).await;
                    my_db.insert_from_search_results(query, neighbors).await;
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
                    ng_cross_compare(&mut player_session, d1i, t1i, d2i, t2i)
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
                    let ds_and_ts = gr_replicated_pairwise_distance(&mut player_session, &pairs)
                        .await
                        .unwrap();
                    let ds_and_ts = gr_to_rep3(&player_session, ds_and_ts).await.unwrap();
                    ng_cross_compare(
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
                    |mut my_db| async move {
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let query = my_db.vector_store.prepare_query(raw_query);
                        let neighbors = my_db.search_to_insert(&query).await;
                        my_db.insert_from_search_results(query, neighbors).await;
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
                    |mut my_db| async move {
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_galois_iris_shares(&mut rng, on_the_fly_query);

                        let query = my_db.vector_store.prepare_query(raw_query);
                        let neighbors = my_db.search_to_insert(&query).await;
                        my_db.is_match(&neighbors).await;
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
