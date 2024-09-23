use aes_prng::AesRng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{db::IrisDB, iris::IrisCode};
use iris_mpc_cpu::{
    database_generators::generate_iris_shares,
    hawkers::{aby3_store::create_ready_made_hawk_searcher, plaintext_store::PlaintextStore},
};
use rand::SeedableRng;

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

fn bench_ready_made_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("ready_made_hnsw");
    group.sample_size(10);

    for database_size in [10000] {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let (_, secret_searcher) = rt.block_on(async move {
            let mut rng = AesRng::seed_from_u64(0_u64);
            create_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap()
        });

        group.bench_function(
            BenchmarkId::new("big-hnsw-insertions", database_size),
            |b| {
                b.to_async(&rt).iter_batched(
                    || secret_searcher.clone(),
                    |mut my_db| async move {
                        let mut rng = AesRng::seed_from_u64(0_u64);
                        let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                        let raw_query = generate_iris_shares(&mut rng, on_the_fly_query);

                        let query = my_db.vector_store.prepare_query(raw_query);
                        let neighbors = my_db.search_to_insert(&query).await;
                        my_db.insert_from_search_results(query, neighbors).await;
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
    bench_ready_made_hnsw
}

criterion_main!(hnsw);
