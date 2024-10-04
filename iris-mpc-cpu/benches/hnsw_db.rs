use aes_prng::AesRng;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use hawk_pack::{
    graph_store::{GraphMem, GraphPg},
    hnsw_db::HawkSearcher,
    DbStore, VectorStore,
};
use iris_mpc_common::iris_db::db::IrisDB;
use iris_mpc_cpu::hawkers::{
    plaintext_store::PlaintextStore, plaintext_store_db::PlaintextStoreDb,
};
use rand::{RngCore, SeedableRng};
use std::vec;

const HAWK_DATABASE_URL: &str = "postgres://postgres:postgres@localhost/postgres";

/// Table names
const HAWK_GRAPH_ENTRY: &str = "hawk_graph_entry";
const HAWK_GRAPH_LINKS: &str = "hawk_graph_links";
const HAWK_VECTORS: &str = "hawk_vectors";

fn csv_filename(db_size: usize, table_name: String) -> String {
    format!("hnsw_db_{}_{}.csv", db_size, table_name)
}

fn zip_filename(db_size: usize, table_name: String) -> String {
    format!("{}.zip", csv_filename(db_size, table_name))
}

fn to_path(file: &str) -> String {
    format!("./benches/assets/{}", file.to_string())
}

fn unzip(zip_file: &str) {
    let zip_path = to_path(zip_file);

    let path = std::path::Path::new(&zip_path);
    let zip = std::fs::File::open(path).unwrap();
    let mut archive: zip::ZipArchive<std::fs::File> = zip::ZipArchive::new(zip).unwrap();
    archive
        .extract("./benches/assets")
        .expect(&format!("Could not extract {}", zip_file));
}

async fn hawk_searcher_from_csv(
    mut rng: impl RngCore,
    db_size: usize,
    graph_store: GraphPg<PlaintextStoreDb>,
    vector_store: PlaintextStoreDb,
) -> HawkSearcher<PlaintextStore, GraphMem<PlaintextStore>> {
    // Unzip hawk_graph_links and hawk_vectors files
    unzip(&zip_filename(db_size, HAWK_VECTORS.to_string()));
    unzip(&zip_filename(db_size, HAWK_GRAPH_LINKS.to_string()));

    let hawk_graph_entry_path = to_path(&csv_filename(db_size, HAWK_GRAPH_ENTRY.to_string()));
    let hawk_graph_links_path = to_path(&csv_filename(db_size, HAWK_GRAPH_LINKS.to_string()));
    let hawk_vectors_path = to_path(&csv_filename(db_size, HAWK_VECTORS.to_string()));

    graph_store
        .copy_in(vec![
            (HAWK_GRAPH_ENTRY.to_string(), hawk_graph_entry_path),
            (HAWK_GRAPH_LINKS.to_string(), hawk_graph_links_path.clone()),
        ])
        .await
        .unwrap();
    std::fs::remove_file(hawk_graph_links_path).unwrap();
    let graph_mem = graph_store.to_graph_mem().await;
    graph_store.cleanup().await.unwrap();

    vector_store
        .copy_in(vec![(HAWK_VECTORS.to_string(), hawk_vectors_path.clone())])
        .await
        .unwrap();
    std::fs::remove_file(hawk_vectors_path).unwrap();
    let vector_mem = vector_store.to_plaintext_store().await;
    vector_store.cleanup().await.unwrap();

    HawkSearcher::new(vector_mem, graph_mem, &mut rng)
}

fn bench_hnsw_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_db");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    for database_size in [100000, 200000] {
        let schema_name = format!("hnsw_db_{}", database_size.to_string());
        let temporary_name = || format!("{}_{}", schema_name, rand::random::<u32>());

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let plain_searcher = rt.block_on(async move {
            let rng = AesRng::seed_from_u64(0_u64);
            let vector_store = PlaintextStoreDb::new(HAWK_DATABASE_URL, &temporary_name())
                .await
                .unwrap();
            let graph_store = GraphPg::new(HAWK_DATABASE_URL, &temporary_name())
                .await
                .unwrap();
            let plain_searcher =
                hawk_searcher_from_csv(rng, database_size, graph_store, vector_store).await;

            plain_searcher
        });

        group.bench_function(BenchmarkId::new("insert", database_size), |b| {
            b.to_async(&rt).iter_batched(
                || plain_searcher.clone(),
                |mut my_db| async move {
                    let mut rng = AesRng::seed_from_u64(0_u64);
                    let on_the_fly_query = IrisDB::new_random_rng(1, &mut rng).db[0].clone();
                    let query = my_db.vector_store.prepare_query(on_the_fly_query.into());
                    let neighbors = my_db.search_to_insert(&query).await;
                    my_db.insert_from_search_results(query, neighbors).await;
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group! {
    hnsw,
    bench_hnsw_db,
}

criterion_main!(hnsw);
