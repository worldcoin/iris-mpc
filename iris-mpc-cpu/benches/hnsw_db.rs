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

/// csv file paths
const HAWK_GRAPH_ENTRY_CSV: &str = "hnsw_db_1000000_69454808_hawk_graph_entry.csv";
const HAWK_GRAPH_LINKS_CSV: &str = "hnsw_db_1000000_69454808_hawk_graph_links.csv";
const HAWK_GRAPH_LINKS_ZIP: &str = "hnsw_db_1000000_69454808_hawk_graph_links.csv.zip";
const HAWK_VECTORS_CSV: &str = "hnsw_db_1000000_3668603835_vectors.csv";
const HAWK_VECTORS_ZIP: &str = "hnsw_db_1000000_3668603835_vectors.csv.zip";

fn to_path(file: &str) -> String {
    format!("./benches/{}", file.to_string())
}

fn unzip(zip_file: &str, csv_file: &str) {
    let zip_path = to_path(zip_file);
    let csv_path = to_path(csv_file);

    let links_path = std::path::Path::new(&zip_path);
    let links_zip = std::fs::File::open(links_path).unwrap();
    let mut archive = zip::ZipArchive::new(links_zip).unwrap();
    let mut outfile = std::fs::File::create(csv_path).unwrap();
    let mut csv = archive.by_name(csv_file).unwrap();
    std::io::copy(&mut csv, &mut outfile).unwrap();
}

async fn hawk_searcher_from_csv(
    mut rng: impl RngCore,
    graph_store: GraphPg<PlaintextStoreDb>,
    vector_store: PlaintextStoreDb,
) -> HawkSearcher<PlaintextStore, GraphMem<PlaintextStore>> {
    // Unzip hawk_graph_links and hawk_vectors files
    unzip(HAWK_GRAPH_LINKS_ZIP, HAWK_GRAPH_LINKS_CSV);
    unzip(HAWK_VECTORS_ZIP, HAWK_VECTORS_CSV);

    let hawk_graph_entry_path = to_path(HAWK_GRAPH_ENTRY_CSV);
    let hawk_graph_links_path = to_path(HAWK_GRAPH_LINKS_CSV);
    let hawk_vectors_path = to_path(HAWK_VECTORS_CSV);

    graph_store
        .copy_in(vec![
            (HAWK_GRAPH_ENTRY.to_string(), hawk_graph_entry_path),
            (HAWK_GRAPH_LINKS.to_string(), hawk_graph_links_path),
        ])
        .await
        .unwrap();
    let graph_mem = graph_store.to_graph_mem().await;
    graph_store.cleanup().await.unwrap();

    vector_store
        .copy_in(vec![(HAWK_VECTORS.to_string(), hawk_vectors_path)])
        .await
        .unwrap();
    let vector_mem = vector_store.to_plaintext_store().await;
    vector_store.cleanup().await.unwrap();

    HawkSearcher::new(vector_mem, graph_mem, &mut rng)
}

fn bench_hnsw_db(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_db");
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

    let database_size = 100000;

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
        let plain_searcher = hawk_searcher_from_csv(rng, graph_store, vector_store).await;

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

    group.finish();
}

criterion_group! {
    hnsw,
    bench_hnsw_db,
}

criterion_main!(hnsw);
