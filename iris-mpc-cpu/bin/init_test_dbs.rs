use std::{error::Error, fs::File, io::BufReader, path::PathBuf};

use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, StoreId, STORE_IDS},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{graph::{graph_store::GraphPg, layered_graph::EntryPoint}, GraphMem, HnswParams, HnswSearcher},
    protocol::shared_iris::GaloisRingSharedIris,
    py_bindings::{limited_iterator, plaintext_store::Base64IrisCode},
};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::izip;
use rand::{RngCore, SeedableRng};
use serde_json::Deserializer;
use tokio::{sync::mpsc, task::JoinSet};
use tracing::info;

// Process input arguments.
#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    #[clap(long = "source")]
    iris_codes_file: PathBuf,

    #[clap(short = 'n')]
    database_size: Option<usize>,

    // TODO support for processing specific index ranges from file

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

// Type: Random number generators used to transform plaintext into secret shares.
type Rngs = (AesRng, AesRng);

// Convertor: Args -> HnswParams.
impl From<&Args> for HnswParams {
    fn from(args: &Args) -> Self {
        let mut params = HnswParams::new(args.ef_constr, args.ef_search, args.M);
        if let Some(q) = args.layer_probability {
            params.layer_probability = q;
        }

        params
    }
}

// Convertor: Args -> Rngs.
impl From<&Args> for Rngs {
    fn from(value: &Args) -> Self {
        (
            AesRng::seed_from_u64(value.hnsw_prng_seed),
            AesRng::seed_from_u64(value.aby3_prng_seed),
        )
    }
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    info!("Initialized tracing subscriber");

    info!("Parsing CLI arguments and initializing in-memory vector and graph stores");

    let args = Args::parse();
    let params = HnswParams::from(&args);
    let (mut hnsw_rng, mut aby3_rng) = Rngs::from(&args);

    info!("Establishing connections with databases");

    let store = iris_mpc_store::Store::new("postgres://postgres:postgres@localhost:5432", "SMPC_dev_2").await.unwrap();
    let graph_pg: GraphPg<PlaintextStore> = GraphPg::from_iris_store(&store);

    // TODO read existing graph and vector stores from database

    info!(
        "Opening NDJSON file of plaintext iris codes: {:?}",
        args.iris_codes_file
    );

    let (tx_l, rx_l) = mpsc::channel::<IrisCode>(256);
    let (tx_r, rx_r) = mpsc::channel::<IrisCode>(256);

    let receivers: BothEyes<_> = [rx_l, rx_r];

    let mut jobs = JoinSet::new();

    tokio::task::spawn_blocking(move || {
        let file = File::open(args.iris_codes_file.as_path()).unwrap();
        let reader = BufReader::new(file);
        let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
        let stream = limited_iterator(stream, args.database_size.map(|x| 2 * x));

        let processors: BothEyes<_> = [tx_l, tx_r];
        for (idx, json_pt) in stream.enumerate() {
            let raw_query = (&json_pt.unwrap()).into();

            let side = idx % 2;
            processors[side].blocking_send(raw_query).unwrap();
        }
    });

    for (side, mut rx) in STORE_IDS.iter().copied().zip(receivers.into_iter()) {
        let params = params.clone();
        let mut hnsw_rng = AesRng::seed_from_u64(hnsw_rng.next_u64());

        jobs.spawn(async move {
            let searcher = HnswSearcher { params };
            let mut vector = PlaintextStore::new();
            let mut graph: GraphMem<PlaintextStore> = GraphMem::new();
            let mut counter = 0usize;

            while let Some(raw_query) = rx.recv().await {
                let query = vector.prepare_query(raw_query);
                searcher
                    .insert(&mut vector, &mut graph, &query, &mut hnsw_rng)
                    .await;

                counter += 1;
                if counter % 1000 == 0 {
                    info!("Processed {} plaintext entries for {} side", counter, side);
                }
            }

            (side, Some(vector), Some(graph))
        });
    }

    info!("Building in-memory plaintext vector stores and HNSW graphs");

    let mut results = jobs.join_all().await;
    results.sort_by_key(|x| x.0 as usize);

    let vectors: BothEyes<_> = [results[0].1.take().unwrap(), results[1].1.take().unwrap()];
    let graphs: BothEyes<_> = [results[0].2.take().unwrap(), results[1].2.take().unwrap()];

    for (graph, side) in izip!(graphs.into_iter(), STORE_IDS) {
        info!(
            "Persisting {} graph over {} iris codes to DB",
            side,
            &vectors[side as usize].points.len()
        );
        persist_graph_db(graph, &graph_pg, side).await?;
    }

    info!("Converting plaintext iris codes locally into secret shares");

    const N_PARTIES: usize = 3;
    for (side, vector) in STORE_IDS.iter().copied().zip(vectors.into_iter()) {
        let database_size = vector.points.len();
        let mut shared_irises: Vec<Vec<_>> = (0..N_PARTIES)
            .map(|_| Vec::with_capacity(database_size))
            .collect();

        info!("Generating {} shares", side);

        for iris in vector.points.iter() {
            let all_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, iris.data.0.clone());
            for (party_id, share) in all_shares.into_iter().enumerate() {
                shared_irises[party_id].push(share);
            }
        }

        // TODO insertion has to be handled with both sides at once as input

        for (party, iris_shares) in shared_irises.into_iter().enumerate() {
            info!("Persisting {} shares for party {} to DB", side, party);
            persist_vector_shares(iris_shares, &store).await?;
        }
    }

    info!("Exited successfully! 🎉");

    Ok(())
}

async fn persist_graph_db(graph: GraphMem<PlaintextStore>, graph_pg: &GraphPg<PlaintextStore>, side: StoreId) -> Result<(), Box<dyn Error>> {
    let mut graph_tx = graph_pg.tx().await.unwrap();

    let GraphMem { entry_point, layers } = graph;

    if let Some( EntryPoint { point, layer } ) = entry_point {
        let mut graph_ops = graph_tx.with_graph(side);
        graph_ops.set_entry_point(point, layer).await;
    }

    for (lc, layer) in layers.into_iter().enumerate() {
        for (idx, (pt, links)) in layer.links.into_iter().enumerate() {
            {
                let mut graph_ops = graph_tx.with_graph(side);
                graph_ops.set_links(pt, links, lc).await;
            }

            if (idx % 1000) == 0 {
                graph_tx.tx.commit().await?;
                graph_tx = graph_pg.tx().await.unwrap();
            }
        }
    }

    graph_tx.tx.commit().await?;

    Ok(())
}


// TODO handle both eyes at once
async fn persist_vector_shares(shares: Vec<GaloisRingSharedIris>, store: &Store) -> Result<(), Box<dyn Error>> {
    let mut tx = store.tx().await?;

    for (idx, iris) in shares.into_iter().enumerate() {
        let GaloisRingSharedIris { code, mask } = iris;

        // Inserting shares and masks in the db. Currently reuses the same
        // share and mask for left and right
        store.insert_irises(
            &mut tx,
            &[StoredIrisRef {
                id: (idx + 1) as i64,
                left_code: &code.coefs,
                left_mask: &mask.coefs,
                right_code: &code.coefs,
                right_mask: &mask.coefs,
            }],
        )
        .await?;

        if (idx % 1000) == 0 {
            tx.commit().await?;
            tx = store.tx().await?;
        }
    }
    tracing::info!("Completed initialization of iris db, committing...");
    tx.commit().await?;
    tracing::info!("Committed");

    Ok(())
}
