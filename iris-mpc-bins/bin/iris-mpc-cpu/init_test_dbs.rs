#![recursion_limit = "256"]

use clap::Parser;
use eyre::{bail, eyre, Result};
use futures::future::try_join_all;
use iris_mpc_common::{
    helpers::smpc_request::{
        IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RECOVERY_UPDATE_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE,
    },
    iris_db::iris::IrisCode,
    postgres::{AccessMode, PostgresClient},
    vector_id::SerialId,
    IrisVectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        load_graphs_from_pg, BothEyes, GraphRef, GraphStore, StoreId, STORE_IDS,
    },
    graph_checkpoint::upload_graph_checkpoint,
    hawkers::aby3::aby3_store::FhdOps,
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef, SharedPlaintextStore},
    hnsw::{
        graph::{
            neighborhood::Neighborhood,
            test_utils::{deserialize_graph, serialize_graph, DbContext},
        },
        searcher::LayerDistribution,
        GraphMem, HnswSearcher, SortedNeighborhood,
    },
    protocol::shared_iris::{GaloisRingSharedIris, GaloisRingSharedIrisPair},
    utils::{constants::N_PARTIES, serialization::types::iris_base64::Base64IrisCode},
};
use itertools::{izip, Itertools};
use rand::{prelude::StdRng, SeedableRng};
use serde_json::Deserializer;
use std::{fs::File, io::BufReader, path::PathBuf, sync::Arc};
use tokio::{
    sync::mpsc,
    task::JoinSet,
    time::{timeout, Duration},
};

/// Default party ordinal identifer.
const DEFAULT_PARTY_IDX: usize = 0;

/// Number of iris code pairs to generate secret shares for at a time.
const SECRET_SHARING_BATCH_SIZE: usize = 5000;

/// Per-operation timeout for remote DB writes, so a wedged tunnel connection
/// fails fast (sqlx has no socket timeout) instead of hanging the whole run.
const SHARE_PERSIST_TIMEOUT: Duration = Duration::from_secs(180);
const GRAPH_PERSIST_TIMEOUT: Duration = Duration::from_secs(1200);

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

    /// Build the graph in linear-scan layer mode (layers capped at
    /// `max_graph_layer`, higher promotions kept as linearly-scanned entry
    /// points), matching the Hawk server's search mode. When false, the
    /// default `Standard` layer mode is used.
    #[clap(long, default_value = "false")]
    linear_scan: bool,

    /// Maximum graph layer for linear-scan mode. Should match the server's
    /// `LINEAR_SCAN_MAX_GRAPH_LAYER`. Ignored unless `--linear-scan` is set.
    #[clap(long, default_value = "1")]
    max_graph_layer: usize,

    /// Number of irises searched in parallel per insertion batch during graph
    /// construction. Larger batches increase search parallelism but reduce
    /// graph quality (same-batch nodes can't link to each other), matching
    /// Hawk Main's batch behavior.
    #[clap(long, default_value = "64")]
    build_batch_size: usize,

    /// Optional path to checkpoint the built in-memory graph. If the file
    /// exists (and matches the persisted iris count) the graph is loaded from
    /// it instead of rebuilt; otherwise the freshly built graph is written
    /// there. Lets a DB-persist failure resume without re-running the build.
    #[clap(long)]
    graph_checkpoint_file: Option<PathBuf>,

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

    /// Migration mode: skip all seeding/graph-build. Instead, for each party load
    /// the existing graph from `hawk_graph_links`, upload it to S3 as a graph
    /// checkpoint, and record the `genesis_graph_checkpoint` row — i.e. produce
    /// the same S3 + DB state a `main`-branch genesis run leaves, without a
    /// re-index. Bootstraps the WAL-branch migration.
    #[clap(long, default_value = "false")]
    migrate_checkpoint: bool,

    /// S3 bucket for each party's graph checkpoint (party order 1, 2, 3).
    /// Required when `--migrate-checkpoint` is set.
    #[clap(long)]
    graph_bucket_party1: Option<String>,
    #[clap(long)]
    graph_bucket_party2: Option<String>,
    #[clap(long)]
    graph_bucket_party3: Option<String>,

    /// AWS region of the graph-checkpoint buckets.
    #[clap(long, default_value = "eu-central-1")]
    graph_bucket_region: String,

    /// Restrict `--migrate-checkpoint` to a single party (0, 1, or 2). Lets each
    /// party's checkpoint be uploaded under that party's own credentials, since
    /// the per-party buckets are pod-IAM-role-scoped. When set, only that party's
    /// `--db-url-party{N+1}` / `--db-schema-party{N+1}` / `--graph-bucket-party{N+1}`
    /// are used; the other slots are ignored (pass any placeholder). When omitted,
    /// all parties are processed in one run.
    #[clap(long)]
    party: Option<usize>,
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
        let mut searcher = if args.linear_scan {
            HnswSearcher::new_linear_scan(args.ef, args.ef, args.M, args.max_graph_layer)
        } else {
            HnswSearcher::new_standard(args.ef, args.ef, args.M)
        };
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

/// Step A of the WAL migration. For each party, load the graph already persisted
/// in `hawk_graph_links`, upload it to that party's S3 bucket as a graph
/// checkpoint (current = v3), and record the `genesis_graph_checkpoint` row —
/// reproducing the S3 + DB state a `main`-branch genesis run leaves, without a
/// re-index. The graph structure is identical across parties, so the three
/// checkpoints share a blake3 hash (what the WAL boot consensus requires).
///
/// Reuses the exact `load_graphs_from_pg` / `upload_graph_checkpoint` /
/// `insert_genesis_graph_checkpoint` paths genesis uses, so the bytes match.
async fn migrate_checkpoint_to_s3(args: &Args) -> Result<()> {
    let buckets = [
        &args.graph_bucket_party1,
        &args.graph_bucket_party2,
        &args.graph_bucket_party3,
    ];
    let urls = args.db_urls();
    let schemas = args.db_schemas();

    let s3_config = aws_config::from_env()
        .region(aws_sdk_s3::config::Region::new(
            args.graph_bucket_region.clone(),
        ))
        .load()
        .await;
    let s3_client = aws_sdk_s3::Client::new(&s3_config);

    let party_ids: Vec<usize> = match args.party {
        Some(p) => {
            if p >= N_PARTIES {
                bail!("--party {p} out of range; must be 0..{N_PARTIES}");
            }
            vec![p]
        }
        None => (0..N_PARTIES).collect(),
    };

    for party_id in party_ids {
        let bucket = buckets[party_id].clone().ok_or_else(|| {
            eyre!(
                "--graph-bucket-party{} is required for --migrate-checkpoint",
                party_id + 1
            )
        })?;
        let schema = &schemas[party_id];
        let url = &urls[party_id];

        tracing::info!("Party {party_id}: connecting to schema {schema}");
        let client = PostgresClient::new(url, schema, AccessMode::ReadWrite).await?;
        let graph_store = GraphStore::new(&client).await?;

        tracing::info!("Party {party_id}: loading graph from hawk_graph_links");
        let [graph_left, graph_right] = load_graphs_from_pg(&graph_store, 8).await?;
        let graph_refs: BothEyes<GraphRef> = [graph_left.to_arc(), graph_right.to_arc()];

        // last_indexed_iris_id: the graph covers every iris in the table. Guard
        // against over-claiming — a checkpoint claiming more coverage than the
        // graph has would make a WAL-branch boot roll back (delete) irises above
        // the real high-water mark. Require graph max node == iris max id.
        let max_iris: i64 = sqlx::query_scalar(&format!(
            "SELECT COALESCE(MAX(id), 0)::bigint FROM \"{schema}\".irises"
        ))
        .fetch_one(graph_store.pool())
        .await?;
        let max_graph_serial: i64 = sqlx::query_scalar(&format!(
            "SELECT COALESCE(MAX(serial_id), 0)::bigint FROM \"{schema}\".hawk_graph_links"
        ))
        .fetch_one(graph_store.pool())
        .await?;
        if max_graph_serial != max_iris {
            bail!(
                "party {party_id}: graph coverage mismatch — hawk_graph_links max serial_id {} != \
                 irises max id {}; refusing to write a checkpoint that misclaims coverage",
                max_graph_serial,
                max_iris
            );
        }

        // last_indexed_modification_id: match genesis exactly — high-water mark
        // over *indexable* modifications only (reset_update, recovery_update,
        // reauth, identity_deletion) that are persisted and COMPLETED. Plain
        // uniqueness enrollments are tracked via last_indexed_iris_id, not here.
        let indexable_types: [&str; 4] = [
            RESET_UPDATE_MESSAGE_TYPE,
            RECOVERY_UPDATE_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE,
            IDENTITY_DELETION_MESSAGE_TYPE,
        ];
        let last_indexed_modification_id: i64 = sqlx::query_scalar::<_, Option<i64>>(&format!(
            "SELECT MAX(id) FROM \"{schema}\".modifications \
             WHERE persisted = true AND status = 'COMPLETED' AND request_type = ANY($1)"
        ))
        .bind(&indexable_types[..])
        .fetch_one(graph_store.pool())
        .await?
        .unwrap_or(0);
        let last_indexed_iris_id: SerialId = max_iris.try_into()?;

        tracing::info!(
            "Party {party_id}: uploading checkpoint to {bucket} \
             (last_indexed_iris_id={last_indexed_iris_id}, \
             last_indexed_modification_id={last_indexed_modification_id})"
        );
        let state = upload_graph_checkpoint(
            &bucket,
            party_id,
            &graph_refs,
            &s3_client,
            last_indexed_iris_id,
            last_indexed_modification_id,
            None, // graph_mutation_id: genesis-style checkpoint
            true, // is_archival
        )
        .await?;

        let mut tx = graph_store.pool().begin().await?;
        GraphStore::insert_genesis_graph_checkpoint(
            &mut tx,
            &state.s3_key,
            last_indexed_iris_id as i64,
            last_indexed_modification_id,
            None,
            &state.blake3_hash,
            true,
            state.graph_version,
        )
        .await?;
        tx.commit().await?;

        tracing::info!(
            "Party {party_id}: checkpoint recorded — s3_key={}, blake3={}, graph_version={}",
            state.s3_key,
            state.blake3_hash,
            state.graph_version
        );
    }

    match args.party {
        Some(p) => tracing::info!("✅ Checkpoint migration complete for party {p}"),
        None => tracing::info!("✅ Checkpoint migration complete for all {N_PARTIES} parties"),
    }
    Ok(())
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    tracing::info!("Initialized tracing subscriber");

    tracing::info!("Parsing CLI arguments");
    let args = Args::parse();

    if args.migrate_checkpoint {
        return migrate_checkpoint_to_s3(&args).await;
    }

    let searcher = HnswSearcher::from(&args);

    tracing::info!("Setting database connections");
    let dbs = init_dbs(&args).await;

    // Reconcile partial/inconsistent state left by an interrupted run: align
    // all parties' iris tables down to the common floor and clear the graphs so
    // they rebuild fresh. Makes any restart safe to resume.
    let n_existing_irises = reconcile_parties(&dbs).await?;
    tracing::info!("Resuming from reconciled iris floor: {}", n_existing_irises);

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

        // Persist the 3 parties concurrently (independent DBs/tunnels), each
        // with a fail-fast timeout so a wedged connection errors out instead of
        // hanging the whole run.
        let persists: Vec<_> = izip!(&dbs, batch.iter_mut())
            .map(|(db, shares)| {
                let shares = std::mem::take(shares);
                async move {
                    let (_, end_serial_id) =
                        timeout(SHARE_PERSIST_TIMEOUT, db.persist_vector_shares(shares))
                            .await
                            .map_err(|_| {
                                eyre!(
                                    "persist_vector_shares timed out after {:?} (wedged tunnel?)",
                                    SHARE_PERSIST_TIMEOUT
                                )
                            })??;
                    if end_serial_id != last_idx {
                        bail!(
                            "serial id mismatch: end={} expected={}",
                            end_serial_id,
                            last_idx
                        );
                    }
                    Ok::<(), eyre::Report>(())
                }
            })
            .collect();
        try_join_all(persists).await?;
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

    // reconcile_parties cleared any partial DB graph, so the graph is always
    // (re)built or loaded from scratch over the full persisted iris set.
    let total_irises = dbs[DEFAULT_PARTY_IDX].store.get_max_serial_id().await?;

    // Load the built graph from a checkpoint file if present and matching; this
    // lets a DB-persist failure resume without re-running the ~O(N) build.
    let loaded = match &args.graph_checkpoint_file {
        Some(p) if p.exists() => {
            let g = deserialize_graph(p).await?;
            let n = get_max_serial_id(&g[0]).unwrap_or(0) as usize;
            if n == total_irises {
                tracing::info!("Loaded graph checkpoint {:?} ({} nodes)", p, n);
                Some(g)
            } else {
                tracing::warn!(
                    "Checkpoint {:?} has {} nodes != {} persisted irises; rebuilding",
                    p,
                    n,
                    total_irises
                );
                None
            }
        }
        _ => None,
    };

    let built = loaded.is_none();
    let graphs: [GraphMem<PlaintextVectorRef>; 2] = if let Some(g) = loaded {
        g
    } else {
        // Graph is empty (cleared), so there is nothing to skip or preload.
        let graphs = [GraphMem::new(), GraphMem::new()];
        let n_existing_irises = 0usize;
        let num_irises = Some(total_irises);
        let vectors = [
            PlaintextStore::<FhdOps>::new(),
            PlaintextStore::<FhdOps>::new(),
        ];

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

        tracing::info!(
            "Initializing per-eye jobs (parallel search, batch size {})",
            args.build_batch_size
        );
        let searcher = Arc::new(searcher);
        for (side, mut rx, vector_store, graph) in izip!(
            STORE_IDS,
            receivers.into_iter(),
            vectors.into_iter(),
            graphs.into_iter(),
        ) {
            let searcher = searcher.clone();
            let prf_seed = (args.hnsw_prf_key as u128).to_le_bytes();
            let skip_serial_ids = args.skip_insert_serial_ids.clone();
            let batch_size = args.build_batch_size;

            jobs.spawn(async move {
                // Arc/RwLock-backed store so each parallel search gets a cheap,
                // shared-read handle (mirrors Hawk Main's per-session store).
                let mut store: SharedPlaintextStore<FhdOps> = vector_store.into();
                let mut graph = graph;
                let mut counter = 0usize;
                let mut closed = false;

                while !closed {
                    // 1. Collect a batch from the channel, assigning insertion
                    //    layers up front (deterministic PRF, as in Hawk Main).
                    let mut batch: Vec<(IrisVectorId, Arc<IrisCode>, usize)> =
                        Vec::with_capacity(batch_size);
                    while batch.len() < batch_size {
                        match rx.recv().await {
                            Some(raw_query) => {
                                let serial_id = raw_query.serial_id;
                                if skip_serial_ids.contains(&serial_id) {
                                    tracing::info!(
                                        "Skipping insertion of serial id {} for {} side",
                                        serial_id,
                                        side
                                    );
                                    continue;
                                }
                                let inserted_id = IrisVectorId::from_serial_id(serial_id);
                                let query = Arc::new(raw_query.iris_code);
                                let insertion_layer =
                                    searcher.gen_layer_prf(&prf_seed, &(inserted_id, side))?;
                                batch.push((inserted_id, query, insertion_layer));
                            }
                            None => {
                                closed = true;
                                break;
                            }
                        }
                    }
                    if batch.is_empty() {
                        break;
                    }

                    // 2. Searches run in parallel against the pre-batch graph
                    //    snapshot; same-batch nodes are invisible to each other,
                    //    matching Hawk Main's batch semantics.
                    let graph_arc = Arc::new(graph);
                    let mut handles = Vec::with_capacity(batch.len());
                    for (_, query, insertion_layer) in &batch {
                        let mut search_store = store.clone();
                        let graph_arc = graph_arc.clone();
                        let searcher = searcher.clone();
                        let query = query.clone();
                        let insertion_layer = *insertion_layer;
                        handles.push(tokio::spawn(async move {
                            let (links, update_ep) = searcher
                                .search_to_insert::<_, SortedNeighborhood<_>>(
                                    &mut search_store,
                                    &graph_arc,
                                    &query,
                                    insertion_layer,
                                )
                                .await?;

                            // Trim each layer's neighborhood to M and extract ids.
                            let mut links_unstructured = Vec::with_capacity(links.len());
                            for (lc, mut l) in links.into_iter().enumerate() {
                                let m = searcher.params.get_M(lc);
                                l.trim(&mut search_store, m).await?;
                                links_unstructured.push(l.edge_ids());
                            }

                            Ok::<_, eyre::Report>(InsertPlanV::<SharedPlaintextStore> {
                                query,
                                links: links_unstructured,
                                update_ep,
                            })
                        }));
                    }

                    let mut plans = Vec::with_capacity(handles.len());
                    for handle in handles {
                        plans.push(Some(handle.await??));
                    }

                    // Reclaim sole ownership of the graph for the insert phase.
                    graph = Arc::try_unwrap(graph_arc)
                        .map_err(|_| eyre!("graph Arc still shared after search tasks"))?;

                    // 3. Apply the batch sequentially via Hawk Main's insert logic.
                    //    Vectors are inserted into the store here; join_plans
                    //    reconciles batch entry-point updates (a no-op in
                    //    linear-scan mode, where multiple appends are all valid).
                    let ids: Vec<Option<IrisVectorId>> =
                        batch.iter().map(|(id, _, _)| Some(*id)).collect();
                    insert::insert(&mut store, &mut graph, &searcher, plans, &ids).await?;

                    counter += batch.len();
                    tracing::info!("Processed {} plaintext entries for {} side", counter, side);
                }

                Ok((side, graph))
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
            get_max_serial_id(&results[0].1).unwrap_or(0)
        );

        let [(.., graph_l), (.., graph_r)] = results.try_into().unwrap();
        [graph_l, graph_r]
    };

    // Checkpoint the freshly built graph so a later DB-persist failure can
    // reload it instead of rebuilding.
    if built {
        if let Some(p) = &args.graph_checkpoint_file {
            tracing::info!("Writing graph checkpoint to {:?}", p);
            serialize_graph(p, &graphs).await?;
        }
    }

    let [graph_l, graph_r] = graphs;

    // Persist to all 3 parties AND both sides concurrently (6-way), each with a
    // fail-fast timeout. Graph topology is public, so the same graph goes to all.
    let graph_l = Arc::new(graph_l);
    let graph_r = Arc::new(graph_r);
    let mut graph_persists = Vec::with_capacity(dbs.len() * 2);
    for (party, db) in dbs.iter().enumerate() {
        for (graph, side) in [
            (graph_l.clone(), StoreId::Left),
            (graph_r.clone(), StoreId::Right),
        ] {
            graph_persists.push(async move {
                tracing::info!("Persisting {} graph for party {}", side, party);
                timeout(
                    GRAPH_PERSIST_TIMEOUT,
                    db.persist_graph_db((*graph).clone(), side),
                )
                .await
                .map_err(|_| {
                    eyre!(
                        "persist_graph_db timed out after {:?} (wedged tunnel?)",
                        GRAPH_PERSIST_TIMEOUT
                    )
                })??;
                Ok::<(), eyre::Report>(())
            });
        }
    }
    try_join_all(graph_persists).await?;

    tracing::info!("Exited successfully! 🎉");

    Ok(())
}

/// Align all parties to a consistent base before (re)indexing: delete iris rows
/// above the common floor (the min serial id across parties) and clear every
/// party's graph so it is rebuilt from scratch. Parties are independent DBs with
/// no cross-party transaction, so an interrupted run can leave them at different
/// counts; this repairs that skew and returns the floor to resume from.
async fn reconcile_parties(dbs: &[DbContext]) -> Result<usize> {
    let mut maxes = Vec::with_capacity(dbs.len());
    for db in dbs {
        maxes.push(db.store.get_max_serial_id().await?);
    }
    let floor = *maxes.iter().min().unwrap_or(&0);
    tracing::info!(
        "Party iris counts: {:?}; reconciling to floor {}",
        maxes,
        floor
    );
    for (party, db) in dbs.iter().enumerate() {
        if maxes[party] > floor {
            tracing::warn!(
                "Party {} has {} irises > floor {}; deleting overflow",
                party,
                maxes[party],
                floor
            );
            db.store.delete_irises_after_id(floor).await?;
        }
        // Always rebuild the graph: clearing here avoids partial-graph skew
        // across parties and is cheap relative to the iris writes.
        db.clear_graph().await?;
    }
    Ok(floor)
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
