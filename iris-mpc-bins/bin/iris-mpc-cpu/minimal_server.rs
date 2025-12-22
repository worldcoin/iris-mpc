use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{
    extract::State,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use clap::{Parser, ValueEnum};
use eyre::{bail, eyre, Result, WrapErr};
use futures::future::try_join_all;
use iris_mpc_common::{
    galois_engine::degree4::{
        preprocess_iris_message_shares, GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
    },
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::{db::IrisDB, iris::IrisCode},
    job::{BatchMetadata, BatchQuery, JobSubmissionHandle},
    test::generate_full_test_db,
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle},
    hawkers::{
        aby3::aby3_store::{Aby3SharedIrises, Aby3Store, Aby3VectorRef},
        build_plaintext::plaintext_parallel_batch_insert,
    },
    hnsw::{GraphMem, HnswSearcher},
    protocol::shared_iris::GaloisRingSharedIris,
    utils::serialization::iris_ndjson::{irises_from_ndjson, IrisSelection},
};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::{fs, net::TcpListener, signal, sync::Mutex};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};
use uuid::Uuid;

/// Delay after request completion before shutting down (allows logs to flush)
const SHUTDOWN_DELAY_MS: u64 = 500;

const DEFAULT_DB_SIZE: usize = 1_000;
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const LINEAR_SCAN_MAX_GRAPH_LAYER: usize = 1;
const DEFAULT_CONTROL_ADDRS: &str = "127.0.0.1:18000,127.0.0.1:18001,127.0.0.1:18002";
const DEMO_TRIGGER_DELAY_SECS: u64 = 5;
/// Batch size for parallel graph insertion (higher = more parallelism but more memory)
const GRAPH_BUILD_BATCH_SIZE: usize = 256;
/// PRF seed for deterministic layer assignment during graph building
const GRAPH_PRF_SEED: [u8; 16] = [0xde, 0xad, 0xbe, 0xef, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum GraphCacheMode {
    Disabled,
    Auto,
    Save,
    Load,
}

#[derive(Serialize, Deserialize)]
struct GraphCacheSnapshot {
    graph: GraphMem<Aby3VectorRef>,
    iris_store: Aby3SharedIrises,
}

#[derive(Parser)]
struct MinimalServerArgs {
    #[clap(flatten)]
    hawk: HawkArgs,

    #[clap(long, value_delimiter = ',', default_value = DEFAULT_CONTROL_ADDRS)]
    control_addrs: Vec<String>,

    #[clap(long, default_value_t = false)]
    initiator: bool,

    #[clap(long, default_value_t = 0)]
    demo_record_index: usize,

    #[clap(long, default_value_t = INTERNAL_RNG_SEED)]
    demo_seed: u64,

    #[clap(long, default_value_t = DEMO_TRIGGER_DELAY_SECS)]
    trigger_delay_secs: u64,

    #[clap(long)]
    graph_cache_path: Option<PathBuf>,

    #[clap(long, value_enum, default_value_t = GraphCacheMode::Auto)]
    graph_cache_mode: GraphCacheMode,

    /// Path to ndjson file containing base64-encoded iris codes.
    #[clap(long)]
    plain_db_ndjson: Option<PathBuf>,

    /// Optional limit on number of iris codes to load from ndjson.
    #[clap(long)]
    plain_db_ndjson_limit: Option<usize>,

    /// Exit after processing a single request (useful for one-shot testing).
    #[clap(long, default_value_t = false)]
    single_request: bool,

    /// Size of the synthetic test database (ignored when using --plain-db-ndjson).
    #[clap(long, default_value_t = DEFAULT_DB_SIZE)]
    db_size: usize,
}

#[derive(Clone, Serialize, Deserialize)]
struct PartyRequestPayload {
    request_id: String,
    sns_id: String,
    skip_persistence: bool,
    code: GaloisRingIrisCodeShare,
    mask: GaloisRingIrisCodeShare,
}

#[derive(Serialize)]
struct JobResponse {
    request_id: String,
}

#[derive(Clone)]
struct ServerState {
    party_index: usize,
    handle: Arc<Mutex<HawkHandle>>,
    shutdown_token: CancellationToken,
    single_request: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();

    let args = MinimalServerArgs::parse();
    info!(
        party_index = args.hawk.party_index,
        "Minimal hawk server starting"
    );
    // Use hawk addresses but with port + 1000 for control HTTP server (to avoid conflict with MPC port)
    let control_addrs: Vec<SocketAddr> = args
        .hawk
        .addresses
        .iter()
        .map(|addr| {
            let mut sock: SocketAddr = addr.parse().expect("invalid hawk address");
            sock.set_port(sock.port() + 1000); // e.g., 16000 -> 17000
            sock
        })
        .collect();

    if args.hawk.party_index >= control_addrs.len() {
        bail!(
            "party_index {} out of range for {} parties",
            args.hawk.party_index,
            control_addrs.len()
        );
    }

    let listen_addr = control_addrs[args.hawk.party_index];
    let control_urls: Vec<String> = control_addrs
        .iter()
        .map(|addr| format!("http://{addr}/job"))
        .collect();

    let plain_db = Arc::new(load_plain_db(&args)?);
    let searcher = HnswSearcher::new_linear_scan(
        args.hawk.hnsw_param_ef_constr,
        args.hawk.hnsw_param_ef_search,
        args.hawk.hnsw_param_M,
        LINEAR_SCAN_MAX_GRAPH_LAYER,
    );
    let (graph, iris_store) = prepare_graph_and_store(&args, &plain_db, &searcher).await?;

    // Duplicate the single graph/store for both eyes (left and right) to work with existing infra
    info!("Creating HawkActor...");
    let hawk_actor = HawkActor::from_cli_with_graph_and_store(
        &args.hawk,
        CancellationToken::new(),
        [graph.clone(), graph],
        [iris_store.clone(), iris_store],
    )
    .await?;
    info!("HawkActor created, now creating HawkHandle (establishing MPC sessions)...");
    let handle = HawkHandle::new(hawk_actor).await?;
    info!("HawkHandle created, MPC sessions established");
    let shutdown_token = CancellationToken::new();
    let state = Arc::new(ServerState {
        party_index: args.hawk.party_index,
        handle: Arc::new(Mutex::new(handle)),
        shutdown_token: shutdown_token.clone(),
        single_request: args.single_request,
    });

    let router = Router::new()
        .route("/job", post(handle_job))
        .with_state(state.clone());

    let listener = TcpListener::bind(listen_addr)
        .await
        .wrap_err("failed to bind control server")?;
    info!("Starting control server on {listen_addr}");
    let server = axum::serve(listener, router.into_make_service());
    tokio::spawn(async move {
        if let Err(err) = server.await {
            error!("control server stopped: {:?}", err);
        }
    });

    if args.initiator {
        let plain_db = Arc::clone(&plain_db);
        let urls = control_urls.clone();
        let demo_index = args.demo_record_index;
        let demo_seed = args.demo_seed;
        let trigger_delay = args.trigger_delay_secs;
        let initiator_shutdown = shutdown_token.clone();
        let single_request = args.single_request;
        tokio::spawn(async move {
            let result =
                trigger_demo_request(&urls, plain_db, demo_index, demo_seed, trigger_delay).await;
            if let Err(err) = &result {
                error!("failed to trigger demo request: {:?}", err);
            }
            if single_request {
                // Give a short delay for logs to flush before signaling shutdown
                tokio::time::sleep(Duration::from_millis(SHUTDOWN_DELAY_MS)).await;
                info!("Single-request mode: initiator signaling shutdown");
                initiator_shutdown.cancel();
            }
        });
    }

    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("Received ctrl+c, shutting down");
        }
        _ = shutdown_token.cancelled() => {
            info!("Shutdown signal received");
        }
    }
    info!("Shutting down minimal server");
    Ok(())
}

async fn handle_job(
    State(state): State<Arc<ServerState>>,
    Json(payload): Json<PartyRequestPayload>,
) -> Response {
    let start_time = std::time::Instant::now();
    let request_id = payload.request_id.clone();
    info!(
        party = state.party_index,
        %request_id,
        sns_id = %payload.sns_id,
        "Received job request"
    );

    let batch = match build_batch_from_payload(&payload) {
        Ok(batch) => batch,
        Err(err) => {
            error!(party = state.party_index, %request_id, "failed to build batch: {:?}", err);
            return (
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid payload: {err}"),
            )
                .into_response();
        }
    };

    info!(
        party = state.party_index,
        %request_id,
        "Submitting batch to hawk"
    );

    let mut handle = state.handle.lock().await;
    let response = handle
        .submit_batch_query(batch)
        .await
        .await
        .map_err(|err| eyre!("hawk job failed: {err:?}"));

    let elapsed = start_time.elapsed();
    let response = match response {
        Ok(result) => {
            let match_count = result.matches.iter().filter(|m| **m).count();
            let total = result.matches.len();
            info!(
                party = state.party_index,
                %request_id,
                matches = match_count,
                total,
                elapsed_ms = elapsed.as_millis(),
                "Batch processed - matches: {}/{}, took {:?}",
                match_count,
                total,
                elapsed
            );
            (axum::http::StatusCode::OK, Json(JobResponse { request_id })).into_response()
        }
        Err(err) => {
            error!(party = state.party_index, %request_id, elapsed_ms = elapsed.as_millis(), "hawk job error after {:?}: {:?}", elapsed, err);
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("hawk job error: {err}"),
            )
                .into_response()
        }
    };

    // Signal shutdown after processing a request in single-request mode
    if state.single_request {
        let shutdown_token = state.shutdown_token.clone();
        tokio::spawn(async move {
            // Small delay to ensure response is sent before shutdown
            tokio::time::sleep(Duration::from_millis(SHUTDOWN_DELAY_MS)).await;
            shutdown_token.cancel();
        });
    }

    response
}

fn build_batch_from_payload(payload: &PartyRequestPayload) -> Result<BatchQuery> {
    let mut batch = BatchQuery::default();
    batch.push_matching_request(
        payload.sns_id.clone(),
        payload.request_id.clone(),
        UNIQUENESS_MESSAGE_TYPE,
        BatchMetadata::default(),
        vec![],
        payload.skip_persistence,
    );

    let mask = GaloisRingTrimmedMaskCodeShare::from(&payload.mask);
    let mask_mirrored = GaloisRingTrimmedMaskCodeShare::from(&payload.mask.mirrored_mask());
    let shares = preprocess_iris_message_shares(
        payload.code.clone(),
        mask,
        payload.code.mirrored_code(),
        mask_mirrored,
    )?;

    // Pass the same shares for both eyes to work with existing infra
    batch.push_matching_request_shares(shares.clone(), shares, true);
    Ok(batch)
}

async fn trigger_demo_request(
    control_urls: &[String],
    plain_db: Arc<IrisDB>,
    record_index: usize,
    demo_seed: u64,
    delay_secs: u64,
) -> Result<()> {
    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
    info!(
        record_index,
        db_size = plain_db.db.len(),
        "Initiator preparing demo request"
    );
    let payloads = build_party_payloads(&plain_db, record_index, demo_seed)?;
    let request_id = payloads[0].request_id.clone();
    info!(%request_id, "Sending demo request to all parties");

    let client = Client::builder().timeout(Duration::from_secs(60)).build()?;

    let futs = control_urls
        .iter()
        .cloned()
        .zip(payloads.into_iter())
        .enumerate()
        .map(|(party_idx, (url, payload))| {
            let client = client.clone();
            let request_id = request_id.clone();
            async move {
                info!(party_idx, %url, %request_id, "Sending request to party");
                let resp = client.post(&url).json(&payload).send().await?;
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                if status.is_success() {
                    info!(party_idx, %status, %body, "Party responded successfully");
                } else {
                    error!(party_idx, %status, %body, "Party responded with error");
                }
                if !status.is_success() {
                    bail!("Party {} returned status {}: {}", party_idx, status, body);
                }
                Result::<_, eyre::Report>::Ok(())
            }
        });

    try_join_all(futs).await?;
    info!(%request_id, "Demo request completed successfully");
    Ok(())
}

fn build_party_payloads(
    plain_db: &IrisDB,
    record_index: usize,
    demo_seed: u64,
) -> Result<Vec<PartyRequestPayload>> {
    if plain_db.db.is_empty() {
        bail!("plain iris database is empty");
    }
    let idx = record_index % plain_db.db.len();
    let iris = plain_db
        .db
        .get(idx)
        .ok_or_else(|| eyre!("missing iris at index {}", idx))?;

    let mut rng = StdRng::seed_from_u64(demo_seed);
    let codes = GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, &mut rng);
    let masks = GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, &mut rng);

    let request_id = Uuid::new_v4().to_string();
    let sns_id = format!("sns-{request_id}");

    Ok((0..3)
        .map(|party| PartyRequestPayload {
            request_id: request_id.clone(),
            sns_id: sns_id.clone(),
            skip_persistence: false,
            code: codes[party].clone(),
            mask: masks[party].clone(),
        })
        .collect())
}

fn install_tracing() {
    use std::io;
    use tracing_subscriber::fmt::writer::MakeWriterExt;

    // Use stderr with ANSI disabled for clean remote output
    let stderr = io::stderr;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(stderr.with_max_level(tracing::Level::TRACE))
        .with_ansi(false)
        .try_init();
}

async fn create_graph_from_plain_db(
    player_index: usize,
    db: &IrisDB,
    searcher: &HnswSearcher,
) -> Result<(GraphMem<Aby3VectorRef>, Aby3SharedIrises)> {
    let db_size = db.len();
    info!(
        db_size,
        batch_size = GRAPH_BUILD_BATCH_SIZE,
        "Building graph and iris shares for {db_size} entries using parallel batch insert"
    );

    // Prepare irises in the format required by plaintext_parallel_batch_insert
    let irises: Vec<(VectorId, IrisCode)> = db
        .db
        .iter()
        .enumerate()
        .map(|(idx, iris)| (VectorId::from_0_index(idx as u32), iris.clone()))
        .collect();

    // Build graph using parallel batch insert (parallelizes search operations within each batch)
    info!("Building HNSW graph with parallel batch insert...");
    let start = std::time::Instant::now();
    let (graph, store) = plaintext_parallel_batch_insert(
        None, // Start with empty graph
        None, // Start with empty store
        irises,
        searcher,
        GRAPH_BUILD_BATCH_SIZE,
        &GRAPH_PRF_SEED,
    )
    .await?;
    info!("HNSW graph built in {:?}", start.elapsed());

    // Generate shares in parallel using rayon
    info!("Generating iris shares in parallel...");
    let start = std::time::Instant::now();

    // Collect iris data for parallel processing (need to read from the locked storage)
    let storage_guard = store.storage.read().await;
    let iris_data: Vec<_> = storage_guard
        .get_sorted_serial_ids()
        .into_iter()
        .map(|serial_id| {
            let vector_id = VectorId::from_serial_id(serial_id);
            let iris = storage_guard
                .get_vector_by_serial_id(serial_id)
                .unwrap()
                .as_ref()
                .clone();
            (serial_id, vector_id, iris)
        })
        .collect();
    drop(storage_guard); // Release lock before parallel processing

    // Generate shares in parallel with deterministic per-entry RNG
    let shared_irises: HashMap<_, _> = iris_data
        .into_par_iter()
        .map(|(serial_id, vector_id, iris)| {
            // Use serial_id combined with base seed for deterministic per-entry RNG
            let entry_seed = DB_RNG_SEED.wrapping_add(serial_id as u64);
            let mut entry_rng = StdRng::seed_from_u64(entry_seed);
            let shares = GaloisRingSharedIris::generate_shares_locally(&mut entry_rng, iris);
            (vector_id, Arc::new(shares[player_index].clone()))
        })
        .collect();

    info!(
        "Iris share generation complete in {:?} ({} shares)",
        start.elapsed(),
        shared_irises.len()
    );

    let iris_store = Aby3Store::new_storage(Some(shared_irises));

    Ok((graph, iris_store))
}

fn load_plain_db(args: &MinimalServerArgs) -> Result<IrisDB> {
    if let Some(path) = &args.plain_db_ndjson {
        load_db_from_ndjson_path(path.as_path(), args.plain_db_ndjson_limit)
    } else {
        let test_db = generate_full_test_db(args.db_size, DB_RNG_SEED, false);
        Ok(test_db.plain_dbs(0).clone())
    }
}

fn load_db_from_ndjson_path(path: &Path, limit: Option<usize>) -> Result<IrisDB> {
    let codes = irises_from_ndjson(path, limit, IrisSelection::All)?;
    Ok(IrisDB { db: codes })
}

async fn prepare_graph_and_store(
    args: &MinimalServerArgs,
    plain_db: &IrisDB,
    searcher: &HnswSearcher,
) -> Result<(GraphMem<Aby3VectorRef>, Aby3SharedIrises)> {
    let build =
        || async { create_graph_from_plain_db(args.hawk.party_index, plain_db, searcher).await };

    match args.graph_cache_mode {
        GraphCacheMode::Disabled => build().await,
        GraphCacheMode::Load => {
            let path = graph_cache_path(args)?;
            info!("Loading graph cache from {}", path.display());
            let snapshot = load_graph_cache(&path).await?;
            Ok((snapshot.graph, snapshot.iris_store))
        }
        GraphCacheMode::Save => {
            let path = graph_cache_path(args)?;
            let (graph, iris_store) = build().await?;
            let snapshot = GraphCacheSnapshot {
                graph: graph.clone(),
                iris_store: iris_store.clone(),
            };
            info!("Writing graph cache to {}", path.display());
            save_graph_cache(&path, &snapshot).await?;
            Ok((graph, iris_store))
        }
        GraphCacheMode::Auto => {
            let path = graph_cache_path(args)?;
            if path.exists() {
                info!("Found existing graph cache at {}", path.display());
                match load_graph_cache(&path).await {
                    Ok(snapshot) => Ok((snapshot.graph, snapshot.iris_store)),
                    Err(err) => {
                        warn!(
                            "Failed to load graph cache at {} ({}); delete or move the file before retrying",
                            path.display(),
                            err
                        );
                        Err(err
                            .wrap_err(format!("graph cache at {} is unreadable", path.display())))
                    }
                }
            } else {
                let (graph, iris_store) = build().await?;
                let snapshot = GraphCacheSnapshot {
                    graph: graph.clone(),
                    iris_store: iris_store.clone(),
                };
                info!("Creating new graph cache at {}", path.display());
                save_graph_cache(&path, &snapshot).await?;
                Ok((graph, iris_store))
            }
        }
    }
}

fn graph_cache_path(args: &MinimalServerArgs) -> Result<PathBuf> {
    if let Some(path) = &args.graph_cache_path {
        return Ok(path.clone());
    }
    let exe = std::env::current_exe().wrap_err("could not determine current executable path")?;
    let dir = exe
        .parent()
        .ok_or_else(|| eyre!("executable path has no parent directory"))?;
    Ok(dir.join("graph.bin"))
}

async fn save_graph_cache(path: &Path, snapshot: &GraphCacheSnapshot) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let data = bincode::serialize(snapshot)?;
    fs::write(path, data).await?;
    Ok(())
}

async fn load_graph_cache(path: &Path) -> Result<GraphCacheSnapshot> {
    let data = fs::read(path).await?;
    Ok(bincode::deserialize(&data)?)
}
