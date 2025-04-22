use std::{fs::File, io::BufReader, path::PathBuf};

use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::{
    iris_db::iris::IrisCode,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_cpu::{
    execution::{
        hawk_main::{session_seeded_rng, StoreId, STORE_IDS},
        session::SessionId,
    },
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{
        graph::{graph_store::GraphPg, layered_graph::EntryPoint},
        vector_store::VectorStoreMut,
        GraphMem, HnswParams, HnswSearcher,
    },
    protocol::shared_iris::GaloisRingSharedIris,
    py_bindings::{
        io::{read_json, write_json},
        limited_iterator,
        plaintext_store::Base64IrisCode,
    },
};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::{izip, Itertools};
use rand_chacha::ChaCha8Rng;

use serde_json::Deserializer;
use tokio::{sync::mpsc, task::JoinSet};
use tracing::{info, warn};

use eyre::Result;

/// Default party ordinal identifer.
const DEFAULT_PARTY_IDX: usize = 0;

/// Build an HNSW graph from plaintext NDJSON encoded iris codes, and persist to
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

    /// PRNG seed for HNSW insertion, used to select the layer at which new
    /// elements are inserted into the hierarchical graph structure.
    #[clap(long, default_value = "0")]
    hnsw_prng_seed: u64,

    /// PRNG seed for ABY3 MPC protocols, used for locally generating secret
    /// shares of iris codes.
    #[clap(default_value = "1")]
    aby3_prng_seed: u64,

    /// Allows for writing only to a db for a specified party. 0-based
    #[clap(long)]
    party_idx: Option<usize>

}

impl Args {
    /// Postgres dB schema names.
    fn db_schemas(&self) -> Vec<String> {
        let v = vec![
            self.db_schema_party1.clone(),
            self.db_schema_party2.clone(),
            self.db_schema_party3.clone(),
        ];

        if let Some(party_idx) = self.party_idx {
            info!("Running for writes only for party id {party_idx}");
            vec![v[party_idx].clone()]
        }else {
            v
        }
    }

    /// Postgres dB server addresses.
    fn db_urls(&self) -> Vec<String> {
        let v = vec![
            self.db_url_party1.clone(),
            self.db_url_party2.clone(),
            self.db_url_party3.clone(),
        ];

        if let Some(party_idx) = self.party_idx {
            vec![v[party_idx].clone()]
        }else {
            v
        }

    }
}

// Type: Random number generators used to transform plaintext into secret shares.
type Rngs = (ChaCha8Rng, ChaCha8Rng, AesRng);

// Convertor: Args -> HnswParams.
impl From<&Args> for HnswParams {
    fn from(args: &Args) -> Self {
        let mut params = HnswParams::new(args.ef, args.ef, args.M);
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
            session_seeded_rng(value.hnsw_prng_seed, StoreId::Left, SessionId(0)),
            session_seeded_rng(value.hnsw_prng_seed, StoreId::Right, SessionId(0)),
            <AesRng as rand::SeedableRng>::seed_from_u64(value.aby3_prng_seed),
        )
    }
}

const N_PARTIES: usize = 1;

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    info!("Initialized tracing subscriber");

    info!("Parsing CLI arguments");
    let args = Args::parse();
    let params = HnswParams::from(&args);

    info!("Setting database connections");
    let dbs = init_dbs(&args).await;

    info!("Setting count of previously indexed irises");
    let n_existing_irises = dbs[DEFAULT_PARTY_IDX].store.get_max_serial_id().await?;
    info!("Found {} existing database irises", n_existing_irises);
    warn!("TODO: Escape if count of persisted irises is not equivalent across all parties");

    // number of iris pairs that need to be inserted to increase the DB size to at least the target
    let num_irises = args
        .target_db_size
        .map(|target| target.saturating_sub(n_existing_irises));

    info!("Setting hnsw pseudo-random number generators");
    let (mut hnsw_rng_l, mut hnsw_rng_r, mut aby3_rng) = Rngs::from(&args);
    let prng_state_filename = args.prng_state_file.into_os_string().into_string().unwrap();
    if n_existing_irises > 0 {
        if let Ok((rng_l, rng_r)) = read_json(&prng_state_filename) {
            info!(
                "Loaded intermediate HNSW PRNG state from file: {}",
                prng_state_filename
            );
            hnsw_rng_l = rng_l;
            hnsw_rng_r = rng_r;
        } else {
            warn!(
                "Couldn't load intermediate HNSW PRNG state from file: {}",
                prng_state_filename
            );
            warn!("Initialized PRNGs from explicit seeds");
        }
    } else {
        info!("Initialized PRNGs from explicit seeds");
    }

    info!("Initializing in-memory vectors from NDJSON file");
    let mut vectors = [PlaintextStore::new(), PlaintextStore::new()];
    if n_existing_irises > 0 {
        let file = File::open(args.path_to_iris_codes.as_path()).unwrap();
        let reader = BufReader::new(file);
        let stream = Deserializer::from_reader(reader)
            .into_iter::<Base64IrisCode>()
            .take(2 * n_existing_irises);

        for (count, json_pt) in stream.enumerate() {
            let raw_query = (&json_pt.unwrap()).into();

            let side = count % 2;
            let query = vectors[side].prepare_query(raw_query);
            vectors[side].insert(&query).await;
        }
    }

    info!("Initializing in-memory graphs from databases");
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

    info!(
        "Reading NDJSON file of plaintext iris codes: {:?}",
        args.path_to_iris_codes
    );

    let (tx_l, rx_l) = mpsc::channel::<IrisCode>(256);
    let (tx_r, rx_r) = mpsc::channel::<IrisCode>(256);
    let processors = [tx_l, tx_r];
    let receivers = [rx_l, rx_r];
    let mut jobs: JoinSet<Result<_>> = JoinSet::new();

    info!("Initializing I/O thread upon which plaintext iris codes will be read");
    let io_thread = tokio::task::spawn_blocking(move || {
        let file = File::open(args.path_to_iris_codes.as_path()).unwrap();
        let reader = BufReader::new(file);

        let stream = Deserializer::from_reader(reader)
            .into_iter::<Base64IrisCode>()
            .skip(2 * n_existing_irises);
        let stream = limited_iterator(stream, num_irises.map(|x| 2 * x));

        let mut count = 0usize;
        for json_pt in stream {
            let raw_query = (&json_pt.unwrap()).into();

            let side = count % 2;
            processors[side].blocking_send(raw_query).unwrap();

            count += 1;
        }

        count
    });

    info!("Initializing jobs to process plaintext iris codes");
    let hnsw_rngs = [hnsw_rng_l, hnsw_rng_r];
    for (side, mut rx, mut vector_store, mut graph, mut hnsw_rng) in izip!(
        STORE_IDS,
        receivers.into_iter(),
        vectors.into_iter(),
        graphs.into_iter(),
        hnsw_rngs.into_iter()
    ) {
        let params = params.clone();

        jobs.spawn(async move {
            let searcher = HnswSearcher { params };
            let mut counter = 0usize;

            while let Some(raw_query) = rx.recv().await {
                let query = vector_store.prepare_query(raw_query);
                searcher
                    .insert(&mut vector_store, &mut graph, &query, &mut hnsw_rng)
                    .await?;

                counter += 1;
                if counter % 1000 == 0 {
                    info!("Processed {} plaintext entries for {} side", counter, side);
                }
            }

            Ok((side, vector_store, graph, hnsw_rng))
        });
    }

    info!("Building in-memory plaintext vector stores and HNSW graphs");

    let mut results: Vec<_> = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;
    results.sort_by_key(|x| x.0 as usize);

    let n_read = io_thread.await?;

    info!(
        "Finished building HNSW graphs with {} nodes",
        results[0].1.points.len()
    );

    let (_, vector_r, graph_r, hnsw_rng_r) = results.remove(1);
    let (_, vector_l, graph_l, hnsw_rng_l) = results.remove(0);

    for (party, db) in dbs.iter().enumerate() {
        for (graph, side) in [(&graph_l, StoreId::Left), (&graph_r, StoreId::Right)] {
            info!("Persisting {} graph for party {}", side, party);
            db.persist_graph_db(graph.clone(), side).await?;
        }
    }

    info!("Converting plaintext iris codes locally into secret shares");

    const BATCH_SIZE: usize = 10_000;
    let mut batch: Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>> = (0..N_PARTIES)
        .map(|_| Vec::with_capacity(BATCH_SIZE))
        .collect();

    for (batch_idx, vectors_batch) in izip!(&vector_l.points, &vector_r.points)
        .skip(n_existing_irises)
        .chunks(BATCH_SIZE)
        .into_iter()
        .enumerate()
    {
        for (left, right) in vectors_batch {
            let left_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, left.data.0.clone());
            let right_shares =
                GaloisRingSharedIris::generate_shares_locally(&mut aby3_rng, right.data.0.clone());

            let left_shares :[GaloisRingSharedIris;1] = [left_shares[args.party_idx.unwrap()].clone()];
            let right_shares :[GaloisRingSharedIris;1] = [right_shares[args.party_idx.unwrap()].clone()];
            for (party, (shares_l, shares_r)) in izip!(left_shares, right_shares).enumerate() {
                batch[party].push((shares_l, shares_r));
            }
        }

        let cur_batch_len = batch[0].len();
        let last_idx = batch_idx * BATCH_SIZE + cur_batch_len + n_existing_irises;

        for (db, shares) in izip!(&dbs, batch.iter_mut()) {
            #[allow(clippy::drain_collect)]
            let (_, end_serial_id) = db.persist_vector_shares(shares.drain(..).collect()).await?;
            assert_eq!(end_serial_id, last_idx);
        }
        info!(
            "Persisted {} locally generated shares",
            last_idx - n_existing_irises
        );
    }

    info!(
        "Finished persisting {} locally generated shares",
        n_read / 2
    );

    info!("Serializing HNSW PRNG intermediate state");
    write_json(&(hnsw_rng_l, hnsw_rng_r), &prng_state_filename)?;

    info!("Exited successfully! 🎉");

    Ok(())
}

async fn init_dbs(args: &Args) -> Vec<DbContext> {
    let mut dbs = Vec::new();

    for (url, schema) in izip!(args.db_urls().iter(), args.db_schemas().iter()).take(N_PARTIES) {
        let client = PostgresClient::new(url, schema, AccessMode::ReadWrite)
            .await
            .unwrap();
        let store = Store::new(&client).await.unwrap();
        let graph_pg = GraphPg::new(&client).await.unwrap();
        dbs.push(DbContext { store, graph_pg });
    }

    dbs
}

struct DbContext {
    /// Postgres store to persist data against
    store: Store,
    graph_pg: GraphPg<PlaintextStore>,
}

impl DbContext {
    async fn persist_graph_db(&self, graph: GraphMem<PlaintextStore>, side: StoreId) -> Result<()> {
        let mut graph_tx = self.graph_pg.tx().await.unwrap();

        let GraphMem {
            entry_point,
            layers,
        } = graph;

        if let Some(EntryPoint { point, layer }) = entry_point {
            let mut graph_ops = graph_tx.with_graph(side);
            graph_ops.set_entry_point(point, layer).await?;
        }

        for (lc, layer) in layers.into_iter().enumerate() {
            for (idx, (pt, links)) in layer.links.into_iter().enumerate() {
                {
                    let mut graph_ops = graph_tx.with_graph(side);
                    graph_ops.set_links(pt, links, lc).await?;
                }

                if (idx % 1000) == 999 {
                    graph_tx.tx.commit().await?;
                    graph_tx = self.graph_pg.tx().await.unwrap();
                }
            }
        }

        graph_tx.tx.commit().await?;

        Ok(())
    }

    /// Extends iris shares table by inserting irises following the current
    /// maximum serial id.
    ///
    /// Returns tuple `(start, end)` giving the first and last serial ids
    /// assigned to the inserted shares.
    async fn persist_vector_shares(
        &self,
        shares: Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>,
    ) -> Result<(usize, usize)> {
        let mut tx = self.store.tx().await?;

        let start_serial_id = self.store.get_max_serial_id().await.unwrap_or(0) + 1;
        let end_serial_id = start_serial_id + shares.len() - 1;

        for batch in &shares.iter().enumerate().chunks(1000) {
            let iris_refs: Vec<_> = batch
                .map(|(idx, (iris_l, iris_r))| StoredIrisRef {
                    id: (start_serial_id + idx) as i64,
                    left_code: &iris_l.code.coefs,
                    left_mask: &iris_l.mask.coefs,
                    right_code: &iris_r.code.coefs,
                    right_mask: &iris_r.mask.coefs,
                })
                .collect();

            self.store.insert_irises(&mut tx, &iris_refs).await?;
            tx.commit().await?;
            tx = self.store.tx().await?;
        }

        Ok((start_serial_id, end_serial_id))
    }

    async fn load_graph_to_mem(
        &self,
        side: StoreId,
    ) -> Result<GraphMem<PlaintextStore>, eyre::Report> {
        let mut graph_tx = self.graph_pg.tx().await.unwrap();
        let mut graph_ops = graph_tx.with_graph(side);

        graph_ops.load_to_mem().await
    }
}
