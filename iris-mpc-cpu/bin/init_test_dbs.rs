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

    const N_PARTIES: usize = 3;
    let urls = [
        ("postgres://postgres:postgres@localhost:5432/SMPC_dev_0", "SMPC_dev_0"),
        ("postgres://postgres:postgres@localhost:5432/SMPC_dev_1", "SMPC_dev_1"),
        ("postgres://postgres:postgres@localhost:5432/SMPC_dev_2", "SMPC_dev_2"),
    ];
    let mut dbs = Vec::new();
    for party in 0..N_PARTIES {
        let (url, schema) = urls[party];
        let store = Store::new(url, schema).await.unwrap();
        dbs.push(DbContext { store });
    }

    // TODO read existing graph and vector stores from database

    info!(
        "Opening NDJSON file of plaintext iris codes: {:?}",
        args.iris_codes_file
    );

    let (tx_l, rx_l) = mpsc::channel::<IrisCode>(256);
    let (tx_r, rx_r) = mpsc::channel::<IrisCode>(256);

    let processors = [tx_l, tx_r];
    let receivers = [rx_l, rx_r];

    let mut jobs = JoinSet::new();

    let io_thread = tokio::task::spawn_blocking(move || {
        let file = File::open(args.iris_codes_file.as_path()).unwrap();
        let reader = BufReader::new(file);
        let stream = Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
        let stream = limited_iterator(stream, args.database_size.map(|x| 2 * x));

        let mut count = 0usize;
        for json_pt in stream {
            let raw_query = (&json_pt.unwrap()).into();

            let side = count % 2;
            processors[side].blocking_send(raw_query).unwrap();

            count += 1;
        }

        count
    });

    for (side, mut rx) in izip!(STORE_IDS, receivers.into_iter()) {
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

            (side, vector, graph)
        });
    }

    info!("Building in-memory plaintext vector stores and HNSW graphs");

    let mut results = jobs.join_all().await;
    results.sort_by_key(|x| x.0 as usize);

    let n_read = io_thread.await?;

    info!("Finished building HNSW graphs with {} nodes", results[0].1.points.len());

    let (_, vector_r, graph_r) = results.remove(1);
    let (_, vector_l, graph_l) = results.remove(0);

    // let vectors = [vector_l, vector_r];
    // let graphs = [graph_l, graph_r];

    // let vectors: BothEyes<_> = [results[0].1.take().unwrap(), results[1].1.take().unwrap()];
    // let graphs: BothEyes<_> = [results[0].2.take().unwrap(), results[1].2.take().unwrap()];


    for (party, db) in dbs.iter().enumerate() {
        for (graph, side) in [(&graph_l, StoreId::Left), (&graph_r, StoreId::Right)] {
            info!(
                "Persisting {} graph for party {}",
                side,
                party
            );
            db.persist_graph_db(graph.clone(), side).await?;
        }
    }

    info!("Converting plaintext iris codes locally into secret shares");

    let batch_size = 10_000usize;
    let mut batch: Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>> = 
        (0..N_PARTIES).map(|_| Vec::with_capacity(batch_size)).collect();

    for (idx, (left, right)) in izip!(vector_l.points, vector_r.points).enumerate() {
        let left_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, left.data.0.clone());
        let right_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, right.data.0.clone());

        for (party, (shares_l, shares_r)) in izip!(left_shares, right_shares).enumerate() {
            batch[party].push((shares_l, shares_r));
        }

        if (idx + 1) % batch_size == 0 {
            for (db, shares) in izip!(&dbs, batch.iter_mut()) {
                db.persist_vector_shares(shares.drain(..).collect()).await?;
            }
            info!("Persisted {} locally generated shares", idx + 1);
        }
    }

    // persist any remaining elements in batch
    for (db, shares) in izip!(&dbs, batch.iter_mut()) {
        db.persist_vector_shares(shares.drain(..).collect()).await?;
    }
    info!("Finished persisting {} locally generated shares", n_read / 2);

    info!("Exited successfully! 🎉");

    Ok(())
}

struct DbContext {
    /// Postgres store to persist data against
    store: Store,
}

impl DbContext {
    async fn persist_graph_db(&self, graph: GraphMem<PlaintextStore>, side: StoreId) -> Result<(), Box<dyn Error>> {
        let graph_pg: GraphPg<PlaintextStore> = GraphPg::from_iris_store(&self.store);
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

                if (idx % 1000) == 999 {
                    graph_tx.tx.commit().await?;
                    graph_tx = graph_pg.tx().await.unwrap();
                }
            }
        }

        graph_tx.tx.commit().await?;

        Ok(())
    }

    /// Extends iris shares table by inserting irises following the current
    /// maximum serial id.
    /// TODO specify starting index, and link to above logic
    async fn persist_vector_shares(&self, shares: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>) -> Result<(), Box<dyn Error>> {
        let mut tx = self.store.tx().await?;

        let last_serial_id = self.store.get_max_serial_id().await.unwrap_or(0);

        for (idx, (iris_l, iris_r)) in shares.into_iter().enumerate() {
            let GaloisRingSharedIris { code: code_l, mask: mask_l } = iris_l;
            let GaloisRingSharedIris { code: code_r, mask: mask_r } = iris_r;

            // Inserting shares and masks in the db. Currently reuses the same
            // share and mask for left and right
            self.store.insert_irises(
                &mut tx,
                &[StoredIrisRef {
                    id: (last_serial_id + idx + 1) as i64,
                    left_code: &code_l.coefs,
                    left_mask: &mask_l.coefs,
                    right_code: &code_r.coefs,
                    right_mask: &mask_r.coefs,
                }],
            )
            .await?;

            if (idx % 1000) == 999 {
                tx.commit().await?;
                tx = self.store.tx().await?;
            }
        }
        tracing::info!("Completed initialization of iris db, committing...");
        tx.commit().await?;
        tracing::info!("Committed");

        Ok(())
    }
}
