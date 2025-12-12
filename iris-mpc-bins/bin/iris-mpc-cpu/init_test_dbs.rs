use clap::Parser;
use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, vector_id::SerialId, IrisVectorId};
use iris_mpc_cpu::{
    execution::hawk_main::{StoreId, STORE_IDS},
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef},
    hnsw::{
        graph::test_utils::DbContext, searcher::LayerDistribution, vector_store::VectorStoreMut,
        GraphMem, HnswSearcher, SortedNeighborhood,
    },
    protocol::shared_iris::{GaloisRingSharedIris, GaloisRingSharedIrisPair},
    utils::{
        constants::N_PARTIES,
        serialization::{
            iris_ndjson::{irises_from_ndjson_iter, IrisSelection},
            types::iris_base64::Base64IrisCode,
        },
    },
};
use itertools::{izip, Itertools};
use rand::{prelude::StdRng, SeedableRng};
use serde_json::Deserializer;
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{sync::mpsc, task::JoinSet};

/// Default party ordinal identifer.
const DEFAULT_PARTY_IDX: usize = 0;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

/// Builds an HNSW graph from plaintext NDJSON encoded iris codes & persists to
/// Postgres databases. This binary iteratively builds up a graph in stages
/// over multiple executions. On initial execution, a new GraphMem and
/// PlaintextStore are constructed and initialized from plaintext iris codes,
/// and then the GraphMem is persisted to each MPC party database, and all iris
/// codes are secret shared and persisted to corresponding MPC party databases.
/// PRNG state for HNSW graph construction is written to a file for future
/// runs.
///
/// On subsequent runs, the previous HNSW graph is loaded from the first party's
/// database (all party graphs should be identical), and corresponding
/// plaintext iris codes are loaded from initial entries in the NDJSON file.
/// (Note: not read from database, to avoid having to reconstitute secret
/// shared data.) The full HNSW graphs for both iris sides are persisted to the
/// MPC party databases, overwriting the previous contents of the corresponding
/// tables, and newly inserted and secret shared irises are appended to the
/// existing iris tables for the MPC parties.
///
/// Additionally, for runs after the first, the binary attempts to read the
/// previous PRNG state for HNSW graph construction from file. This is done for
/// reproducibility, so that a graph that is constructed from a given initial
/// seed in one run of 10,000 entries is identical to that constructed from the
/// same seed in two runs of 5,000 entries.
#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    /// Postgres db schema: party 1.
    #[clap(long)]
    db_schema_party1: String,

    /// Postgres db schema: party 2.
    #[clap(long)]
    db_schema_party2: String,

    /// Postgres db schema: party 3.
    #[clap(long)]
    db_schema_party3: String,

    /// Postgres db server url: party 1.
    /// Example URL format: `postgres://postgres:postgres@localhost:5432`
    #[clap(long)]
    db_url_party1: String,

    /// Postgres db server url: party 2.
    /// Example URL format: `postgres://postgres:postgres@localhost:5432`
    #[clap(long)]
    db_url_party2: String,

    /// Postgres db server url: party 3.
    /// Example URL format: `postgres://postgres:postgres@localhost:5432`
    #[clap(long)]
    db_url_party3: String,

    /// The source file for plaintext iris codes, in NDJSON file format.
    #[clap(long = "source")]
    path_to_iris_codes: PathBuf,

    /// Location of temporary file storing PRNG intermediate state between runs
    /// of this binary.
    #[clap(long, default_value = ".prng_state")]
    prng_state_file: PathBuf,

    /// The target number of left/right iris pairs to build the databases from.
    /// If existing entries are already in the database, then additional entries
    /// are added only to increase the total database size to this value.
    ///
    /// If the persisted databases already have more entries than this number,
    /// then the binary will do nothing, rather than shrinking the databases.
    ///
    /// If this parameter is omitted, then all entries in the source iris code
    /// file will be used.
    #[clap(short('s'), long)]
    target_db_size: Option<usize>,

    // HNSW algorithm parameters
    /// `M` parameter for HNSW insertion.
    ///
    /// Specifies the base size of graph neighborhoods for newly inserted
    /// nodes in the graph.
    #[clap(short, long("hnsw-m"), default_value = "256")]
    M: usize,

    /// `ef` parameter for HNSW insertion.
    ///
    /// Specifies the size of active search neighborhood for insertion layers
    /// during the HNSW insertion process.
    #[clap(long("hnsw-ef"), short, default_value = "320")]
    ef: usize,

    /// The probability that an inserted element is promoted to a higher layer
    /// of the HNSW graph hierarchy.
    #[clap(long("hnsw-p"), short('p'))]
    layer_probability: Option<f64>,

    /// PRF key for HNSW insertion, used to select the layer at which new
    /// elements are inserted into the hierarchical graph structure.
    #[clap(long, default_value = "0")]
    hnsw_prf_key: u64,

    /// Shares seed for iris shares insertion, used to generate secret
    /// shares of iris codes.
    /// The default is 42 to match the default in `shares_encoding.rs`.
    #[clap(long, default_value = "42")]
    iris_shares_rnd: u64,

    /// Skip creation of the HNSW graph. When set to true, only the iris codes
    /// are processed and persisted, without building the HNSW graph.
    #[clap(long, default_value = "false")]
    skip_hnsw_graph: bool,

    /// Skip insertion of specific serial IDs ... used primarily in genesis testing.
    #[clap(long, num_args = 1..)]
    skip_insert_serial_ids: Vec<u32>,
}

impl Args {
    /// Postgres dB schema names.
    fn db_schemas(&self) -> Vec<String> {
        vec![
            self.db_schema_party1.clone(),
            self.db_schema_party2.clone(),
            self.db_schema_party3.clone(),
        ]
    }

    /// Postgres dB server addresses.
    fn db_urls(&self) -> Vec<String> {
        vec![
            self.db_url_party1.clone(),
            self.db_url_party2.clone(),
            self.db_url_party3.clone(),
        ]
    }
}

// Convertor: Args -> HnswSearcher.
impl From<&Args> for HnswSearcher {
    fn from(args: &Args) -> Self {
        let mut searcher = HnswSearcher::new_standard(args.ef, args.ef, args.M);
        if let Some(q) = args.layer_probability {
            match &mut searcher.layer_distribution {
                LayerDistribution::Geometric { layer_probability } => {
                    *layer_probability = q;
                }
            }
        }

        searcher
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IrisCodeWithSerialId {
    pub iris_code: IrisCode,
    pub serial_id: SerialId,
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    tracing::info!("Initialized tracing subscriber");

    tracing::info!("Parsing CLI arguments");
    let args = Args::parse();
    let searcher = HnswSearcher::from(&args);

    tracing::info!("Setting database connections");
    let dbs = init_dbs(&args).await;

    tracing::info!("Setting count of previously indexed irises");
    let n_existing_irises = dbs[DEFAULT_PARTY_IDX].store.get_max_serial_id().await?;
    tracing::info!("Found {} existing database irises", n_existing_irises);
    tracing::warn!(
        "TODO: Escape if count of persisted irises is not equivalent across all parties"
    );

    // number of iris pairs that need to be inserted to increase the DB size to at least the target
    let n_irises = args
        .target_db_size
        .map(|target| target.saturating_sub(n_existing_irises));

    tracing::info!("⚓ ANCHOR: Converting plaintext iris codes locally into secret shares");

    let mut batch: Vec<Vec<GaloisRingSharedIrisPair>> = (0..N_PARTIES)
        .map(|_| Vec::with_capacity(SECRET_SHARING_BATCH_SIZE))
        .collect();
    let mut n_read: usize = 0;

    let file = File::open(args.path_to_iris_codes.as_path()).unwrap();
    let reader = BufReader::new(file);
    let stream = Deserializer::from_reader(reader)
        .into_iter::<Base64IrisCode>()
        .skip(2 * n_existing_irises)
        .map(|x| IrisCode::from(&x.unwrap()))
        .tuples()
        .take(n_irises.unwrap_or(usize::MAX))
        .chunks(SECRET_SHARING_BATCH_SIZE);

    for (batch_idx, vectors_batch) in stream.into_iter().enumerate() {
        let vectors_batch: Vec<(_, _)> = vectors_batch.collect();
        n_read += vectors_batch.len();

        for (left, right) in vectors_batch {
            // Reset RNG for each pair to match shares_encoding.rs behavior
            let mut shares_seed = StdRng::seed_from_u64(args.iris_shares_rnd);

            let left_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, left.clone());
            let right_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, right.clone());
            for (party, (shares_l, shares_r)) in izip!(left_shares, right_shares).enumerate() {
                batch[party].push((shares_l, shares_r));
            }
        }

        let cur_batch_len = batch[0].len();
        let last_idx = batch_idx * SECRET_SHARING_BATCH_SIZE + cur_batch_len + n_existing_irises;

        for (db, shares) in izip!(&dbs, batch.iter_mut()) {
            #[allow(clippy::drain_collect)]
            let (_, end_serial_id) = db.persist_vector_shares(shares.drain(..).collect()).await?;
            assert_eq!(end_serial_id, last_idx);
        }
        tracing::info!(
            "Persisted {} locally generated shares",
            last_idx - n_existing_irises
        );
    }

    tracing::info!("Finished persisting {} locally generated shares", n_read);

    if args.skip_hnsw_graph {
        tracing::info!("Skipping HNSW graph construction");
        return Ok(());
    }

    tracing::info!("⚓ ANCHOR: Constructing HNSW graph databases");

    tracing::info!("Initializing in-memory graphs from databases");
    let graphs = if n_existing_irises > 0 {
        // read graph store from party 0
        let graph_l = dbs[DEFAULT_PARTY_IDX]
            .load_graph_to_mem(StoreId::Left)
            .await?;
        let graph_r = dbs[DEFAULT_PARTY_IDX]
            .load_graph_to_mem(StoreId::Right)
            .await?;
        [graph_l, graph_r]
    } else {
        // new graphs
        [GraphMem::new(), GraphMem::new()]
    };

    let n_existing_irises = {
        let n_left = get_max_serial_id(&graphs[0]).unwrap_or(0);
        let n_right = get_max_serial_id(&graphs[1]).unwrap_or(0);

        assert_eq!(
            n_left, n_right,
            "Max serial id not equal in existing left and right HNSW graphs"
        );
        n_left as usize
    };
    tracing::info!(
        "Detected {} existing irises in database HNSW graphs",
        n_existing_irises
    );

    let num_irises = args
        .target_db_size
        .map(|target| target.saturating_sub(n_existing_irises));

    // TODO: use reader function to read NDJSON file.
    tracing::info!("Initializing in-memory vectors from NDJSON file");
    let mut vectors = [PlaintextStore::new(), PlaintextStore::new()];
    if n_existing_irises > 0 {
        let stream = irises_from_ndjson_iter(
            Path::new(args.path_to_iris_codes.as_path()),
            num_irises,
            IrisSelection::All,
        )?;
        for (count, raw_query) in stream.enumerate() {
            let side = count % 2;
            let query = Arc::new(raw_query);
            vectors[side].insert(&query).await;
        }
    }

    tracing::info!(
        "Reading NDJSON file of plaintext iris codes: {:?}",
        args.path_to_iris_codes
    );

    let (tx_l, rx_l) = mpsc::channel::<IrisCodeWithSerialId>(256);
    let (tx_r, rx_r) = mpsc::channel::<IrisCodeWithSerialId>(256);
    let processors = [tx_l, tx_r];
    let receivers = [rx_l, rx_r];
    let mut jobs: JoinSet<Result<_>> = JoinSet::new();

    tracing::info!("Initializing I/O thread for reading plaintext iris codes");
    let io_thread = tokio::task::spawn_blocking(move || {
        let file = File::open(args.path_to_iris_codes.as_path()).unwrap();
        let reader = BufReader::new(file);

        let stream = Deserializer::from_reader(reader)
            .into_iter::<Base64IrisCode>()
            .skip(2 * n_existing_irises)
            .take(num_irises.map(|x| 2 * x).unwrap_or(usize::MAX));
        for (idx, json_pt) in stream.enumerate() {
            let iris_code_query = (&json_pt.unwrap()).into();
            let serial_id = ((idx / 2) + 1 + n_existing_irises) as u32;
            let raw_query = IrisCodeWithSerialId {
                iris_code: iris_code_query,
                serial_id,
            };

            let side = idx % 2;
            processors[side].blocking_send(raw_query).unwrap();
        }
    });

    tracing::info!("Initializing jobs to process plaintext iris codes");
    for (side, mut rx, mut vector_store, mut graph) in izip!(
        STORE_IDS,
        receivers.into_iter(),
        vectors.into_iter(),
        graphs.into_iter(),
    ) {
        let searcher = searcher.clone();
        let prf_seed = (args.hnsw_prf_key as u128).to_le_bytes();
        let skip_serial_ids = args.skip_insert_serial_ids.clone();

        jobs.spawn(async move {
            let mut counter = 0usize;

            while let Some(raw_query) = rx.recv().await {
                let serial_id = raw_query.serial_id;
                if skip_serial_ids.contains(&serial_id) {
                    tracing::info!(
                        "Skipping insertion of serial id {} for {} side",
                        serial_id,
                        side
                    );
                    continue;
                }
                let query = Arc::new(raw_query.iris_code);

                let inserted_id = IrisVectorId::from_serial_id(serial_id);
                vector_store.insert_with_id(inserted_id, query.clone());

                let insertion_layer = searcher.gen_layer_prf(&prf_seed, &(inserted_id, side))?;
                let (neighbors, update_ep) = searcher
                    .search_to_insert::<_, SortedNeighborhood<_>>(
                        &mut vector_store,
                        &graph,
                        &query,
                        insertion_layer,
                    )
                    .await?;
                searcher
                    .insert_from_search_results(
                        &mut vector_store,
                        &mut graph,
                        inserted_id,
                        neighbors,
                        update_ep,
                    )
                    .await?;

                counter += 1;
                if counter % 1000 == 0 {
                    tracing::info!("Processed {} plaintext entries for {} side", counter, side);
                }
            }

            Ok((side, vector_store, graph))
        });
    }

    tracing::info!("Building in-memory plaintext vector stores and HNSW graphs");

    let mut results: Vec<_> = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;
    results.sort_by_key(|x| x.0 as usize);

    io_thread.await?;

    tracing::info!(
        "Finished building HNSW graphs with {} nodes",
        results[0].1.len()
    );

    let [(.., graph_l), (.., graph_r)] = results.try_into().unwrap();

    for (party, db) in dbs.iter().enumerate() {
        for (graph, side) in [(&graph_l, StoreId::Left), (&graph_r, StoreId::Right)] {
            tracing::info!("Persisting {} graph for party {}", side, party);
            db.persist_graph_db(graph.clone(), side).await?;
        }
    }

    tracing::info!("Exited successfully! 🎉");

    Ok(())
}

async fn init_dbs(args: &Args) -> Vec<DbContext> {
    let mut dbs = Vec::new();
    for (url, schema) in izip!(args.db_urls().iter(), args.db_schemas().iter()).take(N_PARTIES) {
        dbs.push(DbContext::new(url, schema).await);
    }
    dbs
}

fn get_max_serial_id(graph: &GraphMem<PlaintextVectorRef>) -> Option<u32> {
    if let Some(layer) = graph.layers.first() {
        layer
            .get_links_map()
            .keys()
            .map(|vec_id| vec_id.serial_id())
            .max()
    } else {
        None
    }
}
