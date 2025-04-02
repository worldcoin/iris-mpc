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

    // TODO replace with support for serial ranges so external driver script can
    // handle checkpoints
    #[clap(long("checkpoints"), value_delimiter = ',')]
    checkpoints: Vec<usize>,

    // TODO database parameters

    // HNSW algorithm parameters
    // TODO pick appropriate defaults for 2-sided search
    #[clap(short, default_value = "384")]
    M: usize,

    #[clap(long("efc"), default_value = "512")]
    ef_constr: usize,

    #[clap(long("efs"), default_value = "512")]
    ef_search: usize,

    #[clap(short('p'))]
    layer_probability: Option<f64>,

    // PRNG seeds
    #[clap(default_value = "0")]
    hnsw_prng_seed: u64,

    #[clap(default_value = "1")]
    aby3_prng_seed: u64,
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

    let mut hnsw_rng = AesRng::seed_from_u64(args.hnsw_prng_seed);
    let mut aby3_rng = AesRng::seed_from_u64(args.aby3_prng_seed);

    // TODO establish connections with databases

    // TODO read existing graph and vector stores from database
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
            .insert(&mut vector, &mut graph, &query, &mut hnsw_rng)
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

            // Postgres backup -- eg_backup, pg_restore, call from shell
            // Call backups from CLI
            persist_graph_db(&graph);

            // existing graph is read from database
            // index range for build -- from, to
            // test: 0 to 100k
        }
    }

    info!(
        "Persisting final graph at {} iris codes to database",
        counter
    );

    persist_graph_db(&graph);

    info!("Dropping HNSW graph to conserve system memory");
    drop(graph);

    info!("Converting plaintext iris codes locally into secret shares");

    const N_PARTIES: usize = 3;
    let database_size = vector.points.len();
    let mut shared_irises: Vec<Vec<_>> = (0..N_PARTIES)
        .map(|_| Vec::with_capacity(database_size))
        .collect();

    for iris in vector.points.iter() {
        let all_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, iris.data.0.clone());
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

fn persist_graph_db(_graph: &GraphMem<PlaintextStore>) {
    // See: iris_mpc_cpu::hnsw::graph::graph_store::GraphPg.from_iris_store

    // See: iris_mpc_cpu::hnsw::graph::graph_store::GraphOps.set_links
    // See: iris_mpc_cpu::hnsw::graph::graph_store::GraphOps.set_entry_point
}

fn persist_vector_shares(_shares: &[GaloisRingSharedIris]) {
    // See: iris_mpc_store::lib::new
    // With url and schema name, can produce new Store struct using `new`

    // See: iris_mpc_store::lib::insert_irises
}
