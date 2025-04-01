use std::{error::Error, fs::File, io::BufReader, path::PathBuf};

use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{GraphMem, HnswParams, HnswSearcher},
    protocol::shared_iris::GaloisRingSharedIris,
    py_bindings::{limited_iterator, plaintext_store::Base64IrisCode},
};
use rand::SeedableRng;
use serde_json::Deserializer;
use tracing::info;

#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    #[clap(long = "source")]
    iris_codes_file: PathBuf,

    #[clap(short = 'n')]
    database_size: Option<usize>,

    #[clap(long("checkpoints"), value_delimiter = ',')]
    checkpoints: Vec<usize>,

    // HNSW algorithm parameters
    #[clap(short, default_value = "384")]
    M: usize,

    #[clap(long("efc"), default_value = "512")]
    ef_constr: usize,

    #[clap(long("efs"), default_value = "512")]
    ef_search: usize,

    #[clap(short('p'))]
    layer_probability: Option<f64>,
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    info!("Initialized tracing subscriber");

    info!("Parsing CLI arguments and initializing in-memory vector and graph stores");

    let args = Args::parse();

    let M = args.M;
    let ef_constr = args.ef_constr;
    let ef_search = args.ef_search;
    let layer_probability = args.layer_probability;

    let mut params = HnswParams::new(ef_constr, ef_search, M);
    if let Some(q) = layer_probability {
        params.layer_probability = q
    }
    let searcher = HnswSearcher { params };

    let mut rng = AesRng::seed_from_u64(42_u64);
    let mut vector = PlaintextStore::new();
    let mut graph = GraphMem::new();

    info!(
        "Opening NDJSON file of plaintext iris codes: {:?}",
        args.iris_codes_file
    );

    let file = File::open(args.iris_codes_file.as_path()).unwrap();
    let reader = BufReader::new(file);
    let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let stream = limited_iterator(stream, args.database_size);

    info!("Building in-memory plaintext vector store and HNSW graph");

    // Iterate over deserialized objects
    let mut counter = 0usize;

    for json_pt in stream {
        let raw_query = (&json_pt.unwrap()).into();
        let query = vector.prepare_query(raw_query);
        searcher
            .insert(&mut vector, &mut graph, &query, &mut rng)
            .await;

        counter += 1;
        if counter % 1000 == 0 {
            info!("Processed {} plaintext entries", counter);
        }

        if args.checkpoints.contains(&counter) {
            info!(
                "Persisting graph checkpoint at {} iris codes to database",
                counter
            );

            persist_graph_db(&graph);
        }
    }

    info!(
        "Persisting final graph at {} iris codes to database",
        counter
    );

    persist_graph_db(&graph);

    info!("Converting plaintext iris codes locally into secret shares");

    const N_PARTIES: usize = 3;
    let database_size = vector.points.len();
    let mut shared_irises: Vec<Vec<_>> = (0..N_PARTIES)
        .map(|_| Vec::with_capacity(database_size))
        .collect();

    for iris in vector.points.iter() {
        let all_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.data.0.clone());
        for (party_id, share) in all_shares.into_iter().enumerate() {
            shared_irises[party_id].push(share);
        }
    }

    info!("Persisting secret shared iris codes to database");
    for (party, iris_shares) in shared_irises.iter().enumerate() {
        info!("Persisting party {} iris code secret shares", party);
        persist_vector_shares(iris_shares);
    }

    info!("Exited successfully! 🎉");

    Ok(())
}

fn persist_graph_db(_graph: &GraphMem<PlaintextStore>) {}

fn persist_vector_shares(_shares: &[GaloisRingSharedIris]) {}
