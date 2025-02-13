use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::iris_db::db::IrisDB;
use iris_mpc_cpu::{
    database_generators::generate_galois_iris_shares,
    hawkers::aby3_store::Aby3Store,
    hnsw::{metrics::network::NetworkFormatter, GraphMem, HnswSearcher, VectorStore},
};
use rand::{RngCore, SeedableRng};
use std::{error::Error, fs::File};
use tokio::task::JoinSet;
use tracing::{trace_span, Instrument};
use tracing_forest::{tag::NoTag, ForestLayer, PrettyPrinter};
use tracing_subscriber::{filter::filter_fn, layer::SubscriberExt, util::SubscriberInitExt, Layer};

#[derive(Parser)]
struct Args {
    #[clap(short = 'n', default_value = "1000")]
    database_size: usize,
}

async fn insert<V: VectorStore>(
    searcher: HnswSearcher,
    vector_store: &mut V,
    graph_store: &mut GraphMem<V>,
    query: &V::QueryRef,
    rng: &mut impl RngCore,
) -> V::VectorRef {
    searcher.insert(vector_store, graph_store, query, rng).await
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let database_size = args.database_size;

    let file = File::create("searcher_network_tree.txt")?;
    let file_processor = PrettyPrinter::new()
        .formatter(NetworkFormatter {})
        .writer(std::sync::Mutex::new(file));

    tracing_subscriber::registry()
        .with(
            ForestLayer::new(file_processor, NoTag {}).with_filter(filter_fn(|metadata| {
                metadata.target().starts_with("searcher")
            })),
        )
        .init();

    let mut rng = AesRng::seed_from_u64(0_u64);

    let (_, vectors_graphs) = Aby3Store::lazy_setup_from_files_with_grpc(
        "./iris-mpc-cpu/data/store.ndjson",
        &format!("./iris-mpc-cpu/data/graph_{}.dat", database_size),
        &mut rng,
        database_size,
        false,
    )
    .await?;

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

        let party_id = vector_store.owner.clone();
        let party_span = trace_span!(target: "searcher", "party", party = ?party_id);

        jobs.spawn(async move {
            insert(
                searcher,
                &mut vector_store,
                &mut graph_store,
                &query,
                &mut rng,
            )
            .instrument(party_span)
            .await;
        });
    }
    jobs.join_all().await;
    Ok(())
}
