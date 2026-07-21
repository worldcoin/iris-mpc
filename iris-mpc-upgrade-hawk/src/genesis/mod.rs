mod graph_checkpoint;

use ampc_server_utils::{
    get_others_sync_state, init_heartbeat_task, set_node_ready, shutdown_handler::ShutdownHandler,
    start_coordination_server_with_extra_routes, try_get_endpoint_other_nodes,
    wait_for_others_ready, wait_for_others_unready, BatchSyncSharedState, ServerCoordinationConfig,
    TaskMonitor,
};
use aws_config::retry::RetryConfig;
use aws_sdk_rds::Client as RDSClient;
use aws_sdk_s3::{
    config::{Builder as S3ConfigBuilder, Region},
    Client as S3Client,
};
use axum::{routing::get, Router};
use chrono::Utc;
use eyre::{bail, eyre, Report, Result};

use iris_mpc_common::{
    config::{CommonConfig, Config, ENV_PROD, ENV_STAGE},
    helpers::sync::Modification,
    iris_db::get_dummy_shares_for_deletion,
    postgres::{run_migrations, AccessMode, PostgresClient},
    SerialId, VectorId, VersionId,
};
pub use iris_mpc_cpu::genesis::BatchSizeConfig;
use iris_mpc_cpu::{
    execution::hawk_main::state_check::SetHash,
    execution::hawk_main::{
        build_hawk_network_handle,
        iris_worker::IrisWorkerPool,
        worker_pool_initializer::{
            DbLoadParams, LocalWorkerPoolInitializer, WorkerPoolInitializer,
        },
        BothEyes, GraphRef, GraphStore, HawkActor, HawkArgs, HawkOps, StoreId, LEFT, RIGHT,
    },
    genesis::{
        state_accessor::{
            get_iris_deletions, get_iris_modifications, get_last_indexed_iris_id,
            set_last_indexed_iris_id, set_last_indexed_modification_id,
        },
        state_sync::{
            Config as GenesisConfig, SyncResult as GenesisSyncResult, SyncState as GenesisSyncState,
        },
        version_join::{
            compute_version_join, make_repair_plan, versions_per_serial, RepairPlan,
            VersionJoinPlan,
        },
        BatchGenerator, BatchIterator, Handle as GenesisHawkHandle, IndexationError, JobRequest,
        JobResult,
    },
    graph_checkpoint::*,
    hawkers::aby3::aby3_store::{Aby3Store, VectorIdRegistryRef},
    hnsw::{graph::graph_store::GraphPg, GraphMem},
    protocol::shared_iris::ArcIris,
};
use iris_mpc_store::{ExplicitVersionToken, Store as IrisStore, StoredIrisRef};
use itertools::izip;
use std::collections::{HashMap, HashSet};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{
        mpsc::{self, Sender},
        oneshot,
    },
    time::timeout,
};

pub use graph_checkpoint::{reset_to_checkpoint, upload_and_sync_genesis_checkpoint};
pub use iris_mpc_cpu::graph_checkpoint::{
    get_common_checkpoint, get_most_recent_checkpoints, get_others_graph_hashes,
};

pub const PERSIST_DELAY: usize = 16;
const DEFAULT_REGION: &str = "eu-north-1";

// Delta consensus exchange over the coordination server: each node publishes
// its value into a slot served on a GET route and polls the peers' slots.
const DELTA_REPAIR_ROUTE: &str = "/delta-repair";
const DELTA_REPAIR_ENDPOINT: &str = "delta-repair";
const DELTA_TOMBSTONES_ROUTE: &str = "/delta-tombstones";
const DELTA_TOMBSTONES_ENDPOINT: &str = "delta-tombstones";
const DELTA_EXCHANGE_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const DELTA_EXCHANGE_POLL: Duration = Duration::from_millis(500);

type DeltaExchangeSlot = Arc<tokio::sync::RwLock<Option<Vec<u8>>>>;

/// Server-side slots backing the delta consensus routes.
pub struct DeltaExchangeSlots {
    repair: DeltaExchangeSlot,
    tombstones: DeltaExchangeSlot,
}

impl DeltaExchangeSlots {
    fn new() -> Self {
        Self {
            repair: Arc::new(tokio::sync::RwLock::new(None)),
            tombstones: Arc::new(tokio::sync::RwLock::new(None)),
        }
    }
}

/// GET handler for a delta exchange slot: 503 until the local value is
/// published, then its serialized bytes.
fn delta_slot_route(slot: DeltaExchangeSlot) -> axum::routing::MethodRouter {
    get(move || {
        let slot = slot.clone();
        async move {
            match slot.read().await.clone() {
                Some(payload) => (axum::http::StatusCode::OK, payload),
                None => (axum::http::StatusCode::SERVICE_UNAVAILABLE, Vec::new()),
            }
        }
    })
}

/// One exchange round: publish `value` on this node's slot, poll the peers'
/// `endpoint` until both answer, and return their values (peer order as
/// returned by the coordination layer).
///
/// Callers must rendezvous first ([`delta_exchange_barrier`]): the deadline
/// starts at publish, and the work preceding an exchange runs at party-local
/// speed.
async fn exchange_delta_state<T: serde::Serialize + serde::de::DeserializeOwned>(
    server_coord_config: &ServerCoordinationConfig,
    slot: &DeltaExchangeSlot,
    endpoint: &str,
    value: &T,
) -> Result<Vec<T>> {
    *slot.write().await = Some(serde_json::to_vec(value)?);
    let deadline = Instant::now() + DELTA_EXCHANGE_TIMEOUT;
    loop {
        let responses = try_get_endpoint_other_nodes(server_coord_config, endpoint).await?;
        if responses.iter().all(|(status, _)| status.is_success()) {
            return responses
                .into_iter()
                .map(|(_, body)| serde_json::from_slice(&body).map_err(Into::into))
                .collect();
        }
        if Instant::now() > deadline {
            bail!("timed out waiting for peers on {endpoint}");
        }
        tokio::time::sleep(DELTA_EXCHANGE_POLL).await;
    }
}

/// Rendezvous ahead of a timed delta exchange, carrying the shutdown flag; a
/// shutdown on any party aborts the delta (a partial delta must not proceed).
async fn delta_exchange_barrier(
    hawk_handle: &mut GenesisHawkHandle,
    shutdown_handler: &ShutdownHandler,
) -> Result<()> {
    let shutdown = shutdown_handler.is_shutting_down();
    let mismatched = hawk_handle.sync_state(shutdown, None).await?;
    if shutdown || mismatched {
        bail!("shutdown signalled during version-join delta");
    }
    Ok(())
}

/// Process input arguments typically passed from command line.
#[derive(Debug, Clone)]
pub struct ExecutionArgs {
    // Serial identifier of maximum indexed Iris.
    pub max_indexation_id: SerialId,

    // Batch size configuration (static or dynamic with cap).
    pub batch_size_config: BatchSizeConfig,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    pub perform_snapshot: bool,

    // Number of irises to index between checkpoints.
    pub checkpoint_frequency: usize,

    // Controls which older checkpoints are pruned after loading a common checkpoint.
    pub pruning_mode: PruningMode,

    // Pinned base checkpoint blake3 hash; None selects the latest common checkpoint.
    pub base_checkpoint_hash: Option<String>,
}

impl ExecutionArgs {
    // this is for integration tests
    pub fn from_plaintext_args(
        args: iris_mpc_cpu::genesis::plaintext::GenesisArgs,
        perform_snapshot: bool,
    ) -> Self {
        Self {
            max_indexation_id: args.max_indexation_id,
            batch_size_config: args.batch_size_config,
            perform_snapshot,
            checkpoint_frequency: args.checkpoint_frequency,
            pruning_mode: args.pruning_mode,
            base_checkpoint_hash: None,
        }
    }
}

/// Information associated with inner execution context.
struct ExecutionContextInfo {
    /// Process input args.
    args: ExecutionArgs,

    /// Process configuration.
    config: Config,

    // Serial idenitifer of last indexed Iris.
    last_indexed_id: SerialId,

    // Set identifiers of Iris's to be excluded from indexation.
    excluded_serial_ids: Vec<SerialId>,

    // Modifications since the base checkpoint's cursor; comparison-log input only.
    modifications: Vec<Modification>,

    // The largest modification id completed and persisted in the source store.
    // Used to track up to which modification the next run of Genesis can start from
    max_modification_persist_id: i64,

    // Whether a common base checkpoint was found (false = fresh start).
    has_base_checkpoint: bool,
}

/// Constructor.
impl ExecutionContextInfo {
    fn new(
        args: &ExecutionArgs,
        config: &Config,
        last_indexed_id: SerialId,
        excluded_serial_ids: Vec<SerialId>,
        modifications: Vec<Modification>,
        max_modification_persist_id: i64,
        has_base_checkpoint: bool,
    ) -> Self {
        Self {
            args: args.clone(),
            config: config.clone(),
            excluded_serial_ids,
            last_indexed_id,
            modifications,
            max_modification_persist_id,
            has_base_checkpoint,
        }
    }
}

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares
///
/// # Arguments
///
/// * `args` - Process arguments.
/// * `config` - Process configuration instance.
///
pub async fn exec(args: ExecutionArgs, config: Config) -> Result<()> {
    tracing::info!("running genesis with \n {:?} \n {:?}", args, config);

    // Phase 0: setup.
    let (
        ctx,
        shutdown_handler,
        mut task_monitor_bg,
        checkpoint_s3_client,
        aws_rds_client,
        registries,
        worker_pools,
        imem_graph_stores,
        mut hawk_handle,
        tx_results,
        graph_store,
        hnsw_iris_store,
        delta_exchange,
    ) = exec_setup(&args, &config).await?;

    tracing::info!("Setup complete.");
    tracing::info!(
        "Starting Genesis indexing process with the following parameters:\n  Max indexation ID: {}\n  Batch size config: {}\n  Perform snapshot: {}",
        args.max_indexation_id,
        args.batch_size_config,
        args.perform_snapshot,
    );

    // Phase 1: apply delta. A fresh start (no base checkpoint, empty state)
    // has nothing to reconcile; only the modification-cursor stamp is written.
    if ctx.has_base_checkpoint {
        hawk_handle = exec_delta(
            &config,
            &ctx,
            graph_store.clone(),
            &checkpoint_s3_client,
            &registries,
            &worker_pools,
            &hnsw_iris_store,
            &imem_graph_stores,
            &delta_exchange,
            hawk_handle,
            &tx_results,
            &mut task_monitor_bg,
            &shutdown_handler,
        )
        .await?;
        tracing::info!("Delta complete.");
    } else {
        tracing::info!(
            "No base checkpoint; skipping delta. Setting last indexed modification id to {}",
            ctx.max_modification_persist_id
        );
        let mut graph_tx = graph_store.tx().await?;
        set_last_indexed_modification_id(&mut graph_tx.tx, ctx.max_modification_persist_id).await?;
        graph_tx.tx.commit().await?;
    }

    // Phase 2: indexation.
    exec_indexation(
        &ctx,
        &checkpoint_s3_client,
        &registries,
        &worker_pools,
        &imem_graph_stores,
        hawk_handle,
        &tx_results,
        task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    tracing::info!("Indexation complete.");

    // Phase 3: snapshot.
    if !args.perform_snapshot {
        tracing::info!("Snapshot skipped.");
    } else {
        exec_snapshot(&ctx, &aws_rds_client).await?;
        tracing::info!("Snapshot complete.");
    };

    // Clear modifications from the HNSW iris store
    // This is because after a genesis run - there should be no modifications left in the HNSW iris store
    let mut tx = hnsw_iris_store.tx().await?;
    hnsw_iris_store
        .clear_modifications_table(&mut tx)
        .await
        .map_err(|err| {
            let msg = format!("Failed to clear modifications: {:?}", err);
            tracing::error!("{}", msg);
            eyre!(msg)
        })?;
    tx.commit().await?;

    tracing::info!("Cleared modifications from the HNSW iris store");

    // trigger manual shutdown to ensure the health check services terminate
    shutdown_handler.trigger_manual_shutdown();

    Ok(())
}

/// Execute process setup tasks.
///
/// # Arguments
///
/// * `args` - Process arguments.
/// * `config` - Process configuration instance.
///
async fn exec_setup(
    args: &ExecutionArgs,
    config: &Config,
) -> Result<(
    ExecutionContextInfo,
    Arc<ShutdownHandler>,
    TaskMonitor,
    S3Client,
    RDSClient,
    BothEyes<VectorIdRegistryRef>,
    BothEyes<Arc<dyn IrisWorkerPool>>,
    Arc<BothEyes<GraphRef>>,
    GenesisHawkHandle,
    Sender<JobResult>,
    Arc<GraphPg<Aby3Store<HawkOps>>>,
    IrisStore,
    DeltaExchangeSlots,
)> {
    // Bail if config is invalid.
    validate_config(config)?;

    // Set shutdown handler.
    let shutdown_handler = init_shutdown_handler(config).await;

    // Set background task monitor.
    let mut task_monitor_bg = TaskMonitor::new();

    // Set service clients.
    let (
        (aws_s3_client, checkpoint_s3_client, aws_rds_client),
        (iris_store, (hnsw_iris_store, graph_store)),
    ) = get_service_clients(config).await?;
    tracing::info!("Service clients instantiated");

    // Verify checkpoint S3 access before any mutations to fail fast on misconfiguration.
    verify_s3_checkpoint_access(
        &checkpoint_s3_client,
        &config.graph_checkpoint_bucket_name,
        config.party_id,
    )
    .await?;
    tracing::info!("Checkpoint S3 bucket access verified");

    let graph_store_arc = Arc::new(graph_store);

    // Set serial identifier of last indexed Iris.
    let last_indexed_id = get_last_indexed_iris_id(graph_store_arc.clone()).await?;
    tracing::info!(
        "Identifier of last Iris to have been indexed = {}",
        last_indexed_id,
    );

    // Set Iris serial identifiers marked for deletion and thus excluded from indexation.
    let excluded_serial_ids =
        get_iris_deletions(config, &aws_s3_client, args.max_indexation_id).await?;
    tracing::info!(
        "Deletions for exclusion count = {}",
        excluded_serial_ids.len(),
    );

    // Snapshot stamp: the global max persisted+completed source modification
    // id. Read BEFORE the registries/worker pools load, so every modification
    // ≤ stamp is reflected in the loaded content — a safe under-claim.
    let max_modification_id_to_persist = iris_store.get_max_persisted_modification_id().await?;
    tracing::info!(
        "Max persisted+completed source modification id = {}",
        max_modification_id_to_persist,
    );

    // Coordinator: Await coordination server to start.
    let genesis_config = GenesisConfig::new(
        args.batch_size_config,
        excluded_serial_ids.clone(),
        args.max_indexation_id,
        args.base_checkpoint_hash.clone(),
    );
    let my_state = get_sync_state(config, genesis_config).await?;
    tracing::info!("Synchronization state initialised");

    let (mut graph_checkpoints, mut hashes) = get_most_recent_checkpoints(&graph_store_arc).await?;

    // Checkpoint pinning: restrict the local candidate list and the hashes
    // advertised on GRAPH_CHECKPOINT_ROUTE to the single pinned entry, so
    // `get_common_checkpoint` selects it on every party or bails. The row is
    // looked up directly by hash — a pin must resolve from anywhere in
    // checkpoint history, not just the recent-hashes exchange window;
    // cross-party agreement on the pin comes from the sync-config equality on
    // `base_checkpoint_hash`.
    let mut pinned_row_id: Option<i64> = None;
    if let Some(pin) = args.base_checkpoint_hash.as_ref() {
        let row = graph_store_arc
            .get_genesis_graph_checkpoint_by_hash(pin)
            .await?
            .ok_or_else(|| {
                eyre!("pinned base checkpoint hash {pin} not found in genesis_graph_checkpoint")
            })?;
        pinned_row_id = Some(row.id);
        let pinned_cp: GraphCheckpointState = row.try_into()?;
        if let Some(newest) = graph_checkpoints.first() {
            if &newest.blake3_hash != pin {
                tracing::info!(
                    "pinned base checkpoint {pin} is older than the newest checkpoint {}",
                    newest.blake3_hash
                );
            }
        }
        let pinned_hash = blake3::Hash::from_hex(pin.as_bytes())
            .map_err(|e| eyre!("pinned base checkpoint hash {pin} is not valid hex: {e}"))?;
        graph_checkpoints = vec![pinned_cp];
        hashes = [[0u8; 32]; 10];
        hashes[0] = *pinned_hash.as_bytes();
    }

    let batch_sync_shared_state =
        Arc::new(tokio::sync::Mutex::new(BatchSyncSharedState::default()));

    let server_coord_config = &config
        .server_coordination
        .clone()
        .unwrap_or_else(|| panic!("Server coordination config is required for server operation"));

    // Coordinator: await server start.
    let delta_exchange = DeltaExchangeSlots::new();
    let extra_routes = Router::new()
        .route(
            GRAPH_CHECKPOINT_ROUTE,
            get(move || async move { serde_json::to_string(&hashes).unwrap() }),
        )
        .route(
            DELTA_REPAIR_ROUTE,
            delta_slot_route(delta_exchange.repair.clone()),
        )
        .route(
            DELTA_TOMBSTONES_ROUTE,
            delta_slot_route(delta_exchange.tombstones.clone()),
        );

    let (is_ready_flag, verified_peers, my_uuid) = start_coordination_server_with_extra_routes(
        server_coord_config,
        &mut task_monitor_bg,
        &shutdown_handler,
        &my_state,
        Some(batch_sync_shared_state),
        Some(extra_routes),
    )
    .await;
    task_monitor_bg.check_tasks();

    // Coordinator: await network state = UNREADY.
    wait_for_others_unready(server_coord_config, &verified_peers, &my_uuid).await?;
    tracing::info!("Network status = UNREADY");
    // Coordinator: await network state = HEALTHY.
    init_heartbeat_task(server_coord_config, &mut task_monitor_bg, &shutdown_handler).await?;
    task_monitor_bg.check_tasks();
    tracing::info!("Network status = HEALTHY");

    // Coordinator: await network state = SYNCHRONIZED.
    let sync_result = get_sync_result(config, &my_state).await?;
    sync_result.check_synced_state()?;
    tracing::info!("Synchronization checks passed");

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        bail!("Shutdown")
    }

    let graph_checkpoint =
        get_common_checkpoint(server_coord_config, hashes, graph_checkpoints).await?;
    tracing::info!("common graph checkpoint: {:?}", graph_checkpoint);

    // The delta reconciles against a base checkpoint. Without one, the only
    // valid state is a fresh start (nothing indexed yet); the delta phase is
    // skipped and indexation starts from zero.
    if graph_checkpoint.is_none() && last_indexed_id != 0 {
        bail!(
            "no common base checkpoint, but last_indexed_iris_id={last_indexed_id} — corrupt state"
        );
    }

    // don't roll anything back if the checkpoint can not be found
    if let Some(cp) = graph_checkpoint.as_ref() {
        if !s3_key_exists(
            &checkpoint_s3_client,
            &config.graph_checkpoint_bucket_name,
            &cp.s3_key,
        )
        .await?
        {
            bail!(
                "s3 checkpoint not found on AWS: s3://{}/{}",
                config.graph_checkpoint_bucket_name,
                cp.s3_key
            );
        }
    }

    // Networking only: sync_peers + rollback must run before the iris
    // load decides `max_serial_id`.
    let (hawk_args, mut hawk_networking) = build_hawk_networking(config, &shutdown_handler).await?;
    hawk_networking.control_channel().await?.sync().await?;

    if let Some(cp) = graph_checkpoint.as_ref() {
        // Reset all HNSW-schema state to the checkpoint: trim the iris
        // tail, restore cursors, clear the WAL and modifications table.
        reset_to_checkpoint(
            cp,
            &graph_store_arc,
            &hnsw_iris_store,
            last_indexed_id,
            pinned_row_id,
        )
        .await?;
    }

    // update if the iris db was reset
    let last_indexed_id = graph_checkpoint
        .as_ref()
        .map(|x| x.last_indexed_iris_id)
        .unwrap_or(last_indexed_id);

    // Fetch the modification list with the checkpoint cursor (NOT the
    // persistent_state cursor, which hawk-main overwrites from a different
    // id-space). Used only for the comparison log; the join sets drive the
    // actual work.
    let modifications = match graph_checkpoint.as_ref() {
        Some(cp) => {
            let (mods, _) = get_iris_modifications(
                &iris_store,
                cp.last_indexed_modification_id,
                last_indexed_id,
            )
            .await?;
            mods
        }
        None => Vec::new(),
    };

    // Bail if stores are inconsistent.
    validate_consistency_of_stores(config, &iris_store, args.max_indexation_id, last_indexed_id)
        .await?;
    tracing::info!("Store consistency checks OK");

    // Iris and graph load in parallel, then assemble the actor.
    let hawk_actor = init_graph_from_stores(
        config,
        &config.graph_checkpoint_bucket_name,
        &iris_store,
        hawk_args,
        hawk_networking,
        &checkpoint_s3_client,
        Arc::clone(&shutdown_handler),
        args.max_indexation_id as usize,
        graph_checkpoint.clone(),
    )
    .await?;
    task_monitor_bg.check_tasks();
    tracing::info!("HNSW graph initialised from store");

    // do this after obtaining the graph from s3. that way if for some reason it isn't there,
    // the old checkpoints could still be found;
    // if the peers are consistent and stores are consistent, then clean up s3 checkpoints
    if let Some(graph_checkpoint) = graph_checkpoint.as_ref() {
        // `graph_checkpoint` is the startup-agreed common checkpoint, so all
        // parties hold it durably; no extra watermark needed.
        if let Err(e) = cleanup_checkpoints(
            &config.graph_checkpoint_bucket_name,
            &checkpoint_s3_client,
            graph_checkpoint,
            None,
            &graph_store_arc,
            args.pruning_mode,
        )
        .await
        {
            tracing::warn!("failed to clean up old s3 checkpoints: {e}");
        }
    }

    // Coordinator: await network state = ready.
    set_node_ready(is_ready_flag);
    let ct = shutdown_handler.get_network_cancellation_token();
    tokio::select! {
        _ = ct.cancelled() => Err(eyre!("ready check failed")),
        r = wait_for_others_ready(server_coord_config) => r
    }?;
    task_monitor_bg.check_tasks();
    tracing::info!("Network status = READY");

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        bail!("Shutdown")
        // return Ok(());
    }

    let registries = hawk_actor.registries();
    let worker_pools = [
        hawk_actor.worker_pool(StoreId::Left),
        hawk_actor.worker_pool(StoreId::Right),
    ];

    // Save graph store references for S3 checkpointing
    let imem_graph_stores: Arc<BothEyes<_>> = Arc::new([
        hawk_actor.graph_store(StoreId::Left),
        hawk_actor.graph_store(StoreId::Right),
    ]);

    // Set Hawk handle.
    let hawk_handle = GenesisHawkHandle::new(hawk_actor).await?;
    tracing::info!("Hawk handle initialised");

    // Set thread for persisting indexing results to DB.
    let tx_results = get_results_thread(
        worker_pools.clone(),
        hnsw_iris_store.clone(),
        graph_store_arc.clone(),
        &mut task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    task_monitor_bg.check_tasks();

    Ok((
        ExecutionContextInfo::new(
            args,
            config,
            last_indexed_id,
            excluded_serial_ids,
            modifications,
            max_modification_id_to_persist,
            graph_checkpoint.is_some(),
        ),
        shutdown_handler,
        task_monitor_bg,
        checkpoint_s3_client,
        aws_rds_client,
        registries,
        worker_pools,
        imem_graph_stores,
        hawk_handle,
        tx_results,
        graph_store_arc,
        hnsw_iris_store,
        delta_exchange,
    ))
}

/// Apply the version-join delta: reconcile the graph and HNSW iris store to the
/// source by `(serial → version)` comparison against the base checkpoint.
///
/// The base checkpoint has already been loaded into the in-memory graph and the
/// HNSW schema reset to it (see `reset_to_checkpoint`). Computes per-eye plans,
/// unions them across eyes and parties, runs the graph repair via the Hawk
/// handle, and only then writes iris rows.
///
/// Ordering invariant: no HNSW row becomes source-consistent before the
/// repaired graph is durable (the post-delta checkpoint). Row disagreements are
/// repair triggers; writing a row first and crashing before the checkpoint
/// would erase the trigger while the distrusted graph survives. A crash
/// anywhere before the flush leaves rows untouched, so a rerun re-derives the
/// same plan. With no graph change the flush runs immediately (no graph
/// durability at stake).
#[allow(clippy::too_many_arguments)]
async fn exec_delta(
    config: &Config,
    ctx: &ExecutionContextInfo,
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
    s3_client: &S3Client,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: &IrisStore,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    delta_exchange: &DeltaExchangeSlots,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    task_monitor_bg: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<GenesisHawkHandle> {
    // S_cp: the checkpoint's last indexed iris id (ctx.last_indexed_id was set
    // to it during reset). Serials are compared over 1..=S_cp.
    let max_serial = ctx.last_indexed_id;
    // Logged cross-check only; tombstones are detected from source content.
    let excluded: HashSet<SerialId> = ctx.excluded_serial_ids.iter().copied().collect();
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;

    let res: Result<(bool, RepairPlan)> = async {
        // 1. Build the version state per eye and compute per-eye plans.
        //    versions_from_iris_store is a single DB scan (one iris row covers both eyes);
        //    versions_from_source are eye-invariant (one source row covers both
        //    eyes), so the LEFT registry is authoritative.
        let hnsw_versions = get_versions_from_iris_store(hnsw_iris_store, max_serial).await?;
        let source_versions = get_versions_from_source(&registries[LEFT], max_serial).await;

        let mut graph_versions: [HashMap<SerialId, Vec<VersionId>>; 2] =
            [HashMap::new(), HashMap::new()];
        for side in [LEFT, RIGHT] {
            graph_versions[side] = get_versions_from_hnsw_graph(&imem_graph_stores[side]).await;
        }
        let plans: [VersionJoinPlan; 2] = [LEFT, RIGHT].map(|side| {
            compute_version_join(
                &graph_versions[side],
                &source_versions,
                &hnsw_versions,
                max_serial,
            )
        });

        // Local repair set = union across eyes. Only per-eye graph state can
        // differ (the store axis is eye-invariant); asymmetric per-eye damage
        // is repairable — removals stay per-eye, reinsertion is uniform (a
        // harmless refresh for the clean eye).
        log_per_eye_asymmetries(&plans, &graph_versions, &source_versions);
        let mut local_repair: Vec<SerialId> = plans[LEFT]
            .graph_repair
            .iter()
            .chain(plans[RIGHT].graph_repair.iter())
            .copied()
            .collect();
        local_repair.sort_unstable();
        local_repair.dedup();

        // 2. Cross-party repair membership. Every replay is an interactive
        //    MPC job that all parties must submit identically, but the row
        //    axis is party-local — so exchange the local sets and take the
        //    union. A replay on a party whose local state was clean is a
        //    harmless refresh.
        delta_exchange_barrier(&mut hawk_handle, shutdown_handler).await?;
        let peer_sets: Vec<Vec<SerialId>> = exchange_delta_state(
            server_coord_config,
            &delta_exchange.repair,
            DELTA_REPAIR_ENDPOINT,
            &local_repair,
        )
        .await?;
        let repair_serials = union_repair_with_peers(&local_repair, &peer_sets);

        // 3. Tombstones among repair serials, by content check against the
        //    party's dummy shares. The classification gates the reinsert flag,
        //    so parties must agree; a mismatch means source-level content
        //    inconsistency, which the repair must not paper over.
        let deleted =
            gather_tombstones(config.party_id, &repair_serials, registries, worker_pools).await?;
        let deleted_hash = {
            let mut h = SetHash::default();
            for s in &deleted {
                h.add_unordered(*s);
            }
            h.checksum()
        };
        delta_exchange_barrier(&mut hawk_handle, shutdown_handler).await?;
        let peer_hashes: Vec<u64> = exchange_delta_state(
            server_coord_config,
            &delta_exchange.tombstones,
            DELTA_TOMBSTONES_ENDPOINT,
            &deleted_hash,
        )
        .await?;
        if peer_hashes.iter().any(|h| *h != deleted_hash) {
            bail!(
                "tombstone sets differ across parties (mine={deleted_hash}, peers={peer_hashes:?}, \
                 count={}); source content is inconsistent",
                deleted.len()
            );
        }

        let serials_in_graph: HashSet<SerialId> = graph_versions[LEFT]
            .keys()
            .chain(graph_versions[RIGHT].keys())
            .copied()
            .collect();
        let repair = make_repair_plan(
            &repair_serials,
            &plans[LEFT].missing_hnsw_rows,
            &deleted,
            &serials_in_graph,
            &source_versions,
            &hnsw_versions,
        );

        // 4. Comparison log (join vs modifications, deletion cross-checks) + metrics.
        log_version_join_comparison(
            &plans,
            &repair_serials,
            &repair,
            &deleted,
            &excluded,
            &ctx.modifications,
            ctx.max_modification_persist_id,
        );

        // 5. Graph repair: replays and removals in serial order, one job per
        //    serial, syncing periodically. Removing an entry point is fine
        //    (the searcher falls back to a temporary EP). No row writes here.
        let mut items: Vec<(SerialId, bool)> = repair
            .graph_replay
            .iter()
            .map(|s| (*s, true))
            .chain(repair.graph_remove.iter().map(|s| (*s, false)))
            .collect();
        items.sort_unstable_by_key(|(serial, _)| *serial);

        if !items.is_empty() {
            metrics::gauge!("genesis_number_version_replays").set(items.len() as f64);
            let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
            let end = items.len().saturating_sub(1);
            let mut now = Instant::now();
            for (idx, (serial, reinsert)) in items.iter().enumerate() {
                // Version lists follow hash iteration order; mutation order
                // must match across parties.
                let removals = [LEFT, RIGHT].map(|side| {
                    let mut keys: Vec<VectorId> = graph_versions[side]
                        .get(serial)
                        .map(|versions| {
                            versions
                                .iter()
                                .map(|v| VectorId::new(*serial, *v))
                                .collect()
                        })
                        .unwrap_or_default();
                    keys.sort_unstable_by_key(|k| k.version_id());
                    keys
                });
                let request = JobRequest::new_version_replay(*serial, removals, *reinsert);
                let result_future = hawk_handle.submit_request(request).await;
                let result =
                    timeout(processing_timeout, result_future)
                        .await
                        .map_err(|err| {
                            tracing::error!("HawkActor processing timeout: {:?}", err);
                            eyre!("HawkActor processing timeout: {:?}", err)
                        })??;

                let (done_rx, result) = result;
                // Increment before send: the results thread decrements on
                // completion, and send-first would let the counter wrap.
                shutdown_handler.increment_batches_pending_completion();
                tx_results.send(result).await?;

                let is_sync_batch = idx % (PERSIST_DELAY - 1) == 0 || idx == end;
                if is_sync_batch {
                    let wait_start = Instant::now();
                    let shutdown = shutdown_handler.is_shutting_down();
                    let mismatched = hawk_handle.sync_state(shutdown, Some(done_rx)).await?;
                    if shutdown || mismatched {
                        // Shutdown on any party: a partial delta must not
                        // proceed to the checkpoint or the row flush.
                        bail!("shutdown signalled during version-join delta");
                    }
                    metrics::histogram!("genesis_persist_wait_duration")
                        .record(wait_start.elapsed().as_secs_f64());
                }
                metrics::histogram!("genesis_version_replay_total_duration",
                    "synced" => if is_sync_batch { "true" } else { "false" })
                .record(now.elapsed().as_secs_f64());
                now = Instant::now();
            }
        }

        Ok((!items.is_empty(), repair))
    }
    .await;

    match res {
        Ok((graph_changed, repair)) => {
            tracing::info!("Waiting for version replays to be processed...");
            let _ = shutdown_handler.wait_for_pending_batches_completion().await;
            tracing::info!("All version replays have been processed");

            // Single end-of-delta cursor write (no per-replay cursor writes):
            // set to the global max persisted+completed source modification id
            // read before the pools loaded (safe under-claim).
            tracing::info!(
                "Setting last indexed modification id to {}",
                ctx.max_modification_persist_id
            );
            let mut graph_tx = graph_store.tx().await?;
            set_last_indexed_modification_id(&mut graph_tx.tx, ctx.max_modification_persist_id)
                .await?;
            graph_tx.tx.commit().await?;

            // S3 checkpoint after delta, only if the graph changed (row writes
            // do not touch the graph, so the base checkpoint still represents it).
            if graph_changed {
                tracing::info!("Creating S3 checkpoint after version-join delta...");
                upload_and_sync_genesis_checkpoint(
                    &config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    ctx.last_indexed_id,
                    ctx.max_modification_persist_id,
                    true, // is_archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                tracing::info!("S3 checkpoint created after version-join delta");
            }

            // Row writes last: the graph repair is durable, so erasing the
            // row-level repair triggers is now safe.
            flush_row_writes(&repair, registries, worker_pools, hnsw_iris_store).await?;

            Ok(hawk_handle)
        }
        Err(err) => {
            tracing::error!(
                "HawkActor processing error while applying version-join delta: {:?}",
                err
            );
            tracing::info!("Initiating shutdown");
            drop(hawk_handle);
            task_monitor_bg.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor_bg.check_tasks_finished();
            Err(err)
        }
    }
}

/// Scan the HNSW iris store's `(serial → version)` index over `1..=max_serial`.
async fn get_versions_from_iris_store(
    hnsw_iris_store: &IrisStore,
    max_serial: SerialId,
) -> Result<HashMap<SerialId, i16>> {
    use futures::TryStreamExt;
    let rows: Vec<(i64, i16)> = hnsw_iris_store
        .stream_iris_ids(max_serial as usize)
        .try_collect()
        .await?;
    Ok(rows
        .into_iter()
        .map(|(id, version)| (id as SerialId, version))
        .collect())
}

/// Build the graph's per-serial key list from layer-0 node keys (layer 0 holds
/// every node, so this enumerates all graph keys).
async fn get_versions_from_hnsw_graph(graph_ref: &GraphRef) -> HashMap<SerialId, Vec<VersionId>> {
    let graph = graph_ref.read().await;
    if graph.layers.is_empty() {
        return HashMap::new();
    }
    versions_per_serial(
        graph.layers[0]
            .links
            .keys()
            .map(|v| (v.serial_id(), v.version_id())),
    )
}

/// Serials whose source content equals the party's deletion dummy on both
/// eyes.
async fn gather_tombstones(
    party_id: usize,
    serials: &[SerialId],
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
) -> Result<HashSet<SerialId>> {
    let mut out = HashSet::new();
    if serials.is_empty() {
        return Ok(out);
    }
    let (dummy_code, dummy_mask) = get_dummy_shares_for_deletion(party_id);
    const CHUNK: usize = 1024;

    for chunk in serials.chunks(CHUNK) {
        let maybe_vids = registries[LEFT].get_vector_ids(chunk).await;
        let mut vids: Vec<VectorId> = Vec::with_capacity(chunk.len());
        for (serial, maybe) in chunk.iter().zip(maybe_vids) {
            let vid = maybe.ok_or_else(|| eyre!("repair serial {serial} missing from registry"))?;
            vids.push(vid);
        }

        let (left_data, right_data) = tokio::try_join!(
            worker_pools[LEFT].fetch_irises(vids.clone()),
            worker_pools[RIGHT].fetch_irises(vids.clone()),
        )?;

        for (i, vid) in vids.iter().enumerate() {
            if left_data[i].code.coefs == dummy_code.coefs
                && left_data[i].mask.coefs == dummy_mask.coefs
                && right_data[i].code.coefs == dummy_code.coefs
                && right_data[i].mask.coefs == dummy_mask.coefs
            {
                out.insert(vid.serial_id());
            }
        }
    }
    Ok(out)
}

/// Build the source `(serial → version)` map from the registry (no DB scan).
async fn get_versions_from_source(
    registry: &VectorIdRegistryRef,
    max_serial: SerialId,
) -> HashMap<SerialId, i16> {
    let reg = registry.read().await;
    let mut out = HashMap::new();
    for serial in 1..=max_serial {
        if let Some(version) = reg.get_current_version(serial) {
            out.insert(serial, version);
        }
    }
    out
}

/// Log serials surgeried in one eye but not the other, with the eye-local
/// graph class. Sample capped.
fn log_per_eye_asymmetries(
    plans: &[VersionJoinPlan; 2],
    graph_versions: &[HashMap<SerialId, Vec<VersionId>>; 2],
    source_versions: &HashMap<SerialId, VersionId>,
) {
    const SAMPLE: usize = 50;

    let graph_class = |side: usize, serial: SerialId| -> &'static str {
        let Some(&v_src) = source_versions.get(&serial) else {
            return "source_missing";
        };
        match graph_versions[side].get(&serial) {
            None => "graph_missing",
            Some(versions) => {
                let v_max = *versions.iter().max().expect("non-empty");
                if v_max < v_src {
                    "version_behind"
                } else if v_max > v_src {
                    "version_ahead"
                } else if versions.len() > 1 {
                    "multi_version"
                } else {
                    "clean"
                }
            }
        }
    };

    for (side, name) in [(LEFT, "left"), (RIGHT, "right")] {
        let other: HashSet<SerialId> = plans[1 - side].graph_repair.iter().copied().collect();
        let only: Vec<(SerialId, &'static str)> = plans[side]
            .graph_repair
            .iter()
            .filter(|s| !other.contains(s))
            .map(|&s| (s, graph_class(side, s)))
            .collect();
        if !only.is_empty() {
            tracing::warn!(
                "version-join: {} serials surgeried only in the {} eye, sample={:?}",
                only.len(),
                name,
                &only[..only.len().min(SAMPLE)],
            );
        }
    }
}

/// Log the join-gated vs modification-gated sets, deletion cross-checks, and
/// per-class gauges. The set hashes cross-check the locally derived sets
/// (tombstones, row repairs), which graph checksums do not cover.
fn log_version_join_comparison(
    plans: &[VersionJoinPlan; 2],
    repair_serials: &[SerialId],
    repair: &RepairPlan,
    deleted: &HashSet<SerialId>,
    excluded: &HashSet<SerialId>,
    modifications: &[Modification],
    max_modification_persist_id: i64,
) {
    const SAMPLE: usize = 50;

    let join_set: HashSet<SerialId> = repair_serials.iter().copied().collect();
    let mod_set: HashSet<SerialId> = modifications
        .iter()
        .filter_map(|m| m.serial_id.map(|s| s as SerialId))
        .collect();

    let intersection = join_set.intersection(&mod_set).count();
    let mut mod_only: Vec<SerialId> = mod_set.difference(&join_set).copied().collect();
    let mut join_only: Vec<SerialId> = join_set.difference(&mod_set).copied().collect();
    let mut mod_all: Vec<SerialId> = mod_set.iter().copied().collect();
    mod_only.sort_unstable();
    join_only.sort_unstable();
    mod_all.sort_unstable();

    // Listed ∧ live = reinserted after deletion (replayed); tombstone ∉ list =
    // deletion the list missed.
    let listed_live: Vec<SerialId> = repair
        .graph_replay
        .iter()
        .filter(|s| excluded.contains(s))
        .copied()
        .collect();
    let mut unlisted_tombstones: Vec<SerialId> = deleted
        .iter()
        .filter(|s| !excluded.contains(s))
        .copied()
        .collect();
    unlisted_tombstones.sort_unstable();

    let hash_of = |serials: &[SerialId]| -> u64 {
        let mut h = SetHash::default();
        for s in serials {
            h.add_unordered(*s);
        }
        h.checksum()
    };
    let sample = |v: &[SerialId]| v[..v.len().min(SAMPLE)].to_vec();

    tracing::info!(
        "version-join comparison: repair={} reasons_left={:?} reasons_right={:?} replay={} \
         remove={} row_inserts={} tombstone_overwrites={} mod_set={} intersection={} mod_only={} \
         join_only={} source_missing={} listed_live={} mod_cursor_stamp={} | replay_hash={} \
         remove_hash={} mod_hash={} row_insert_hash={} tombstone_overwrite_hash={} | \
         mod_only_sample={:?} join_only_sample={:?} row_insert_sample={:?} \
         tombstone_overwrite_sample={:?}",
        repair_serials.len(),
        plans[LEFT].repair_reasons,
        plans[RIGHT].repair_reasons,
        repair.graph_replay.len(),
        repair.graph_remove.len(),
        repair.insert_missing_rows.len(),
        repair.stale_tombstone_rows.len(),
        mod_set.len(),
        intersection,
        mod_only.len(),
        join_only.len(),
        plans[LEFT].source_missing.len(),
        listed_live.len(),
        max_modification_persist_id,
        hash_of(&repair.graph_replay),
        hash_of(&repair.graph_remove),
        hash_of(&mod_all),
        hash_of(&repair.insert_missing_rows),
        hash_of(&repair.stale_tombstone_rows),
        sample(&mod_only),
        sample(&join_only),
        sample(&repair.insert_missing_rows),
        sample(&repair.stale_tombstone_rows),
    );

    if !plans[LEFT].source_missing.is_empty() {
        tracing::error!(
            "version-join: {} serials absent from source pool, sample={:?}",
            plans[LEFT].source_missing.len(),
            sample(&plans[LEFT].source_missing),
        );
    }
    if !unlisted_tombstones.is_empty() {
        tracing::warn!(
            "version-join: {} tombstones not on the S3 deletion list, sample={:?}",
            unlisted_tombstones.len(),
            sample(&unlisted_tombstones),
        );
    }

    metrics::gauge!("genesis_version_join_graph_replay_count")
        .set(repair.graph_replay.len() as f64);
    metrics::gauge!("genesis_version_join_graph_remove_count")
        .set(repair.graph_remove.len() as f64);
    metrics::gauge!("genesis_version_join_row_insert_count")
        .set(repair.insert_missing_rows.len() as f64);
    metrics::gauge!("genesis_version_join_tombstone_overwrite_count")
        .set(repair.stale_tombstone_rows.len() as f64);
    metrics::gauge!("genesis_version_join_mod_only_count").set(mod_only.len() as f64);
    metrics::gauge!("genesis_version_join_join_only_count").set(join_only.len() as f64);
    metrics::gauge!("genesis_version_join_anomaly_count")
        .set(plans[LEFT].source_missing.len() as f64);
}

/// Union the local repair set with the peers', logging per-peer differences
/// (a non-empty difference = party-local row damage). Output sorted.
fn union_repair_with_peers(local: &[SerialId], peer_sets: &[Vec<SerialId>]) -> Vec<SerialId> {
    const SAMPLE: usize = 50;
    let local_set: HashSet<SerialId> = local.iter().copied().collect();
    let mut union: Vec<SerialId> = local.to_vec();
    for (i, peers) in peer_sets.iter().enumerate() {
        let peer_set: HashSet<SerialId> = peers.iter().copied().collect();
        let peer_only: Vec<SerialId> = peers
            .iter()
            .filter(|s| !local_set.contains(s))
            .copied()
            .collect();
        let local_only: Vec<SerialId> = local
            .iter()
            .filter(|s| !peer_set.contains(s))
            .copied()
            .collect();
        if !peer_only.is_empty() || !local_only.is_empty() {
            tracing::warn!(
                "version-join: repair set differs from peer {}: peer_only={} local_only={} \
                 peer_only_sample={:?} local_only_sample={:?}",
                i,
                peer_only.len(),
                local_only.len(),
                &peer_only[..peer_only.len().min(SAMPLE)],
                &local_only[..local_only.len().min(SAMPLE)],
            );
        }
        union.extend(peer_only);
    }
    union.sort_unstable();
    union.dedup();
    union
}

/// Resolve current vector ids and fetch both eyes' pool content for `serials`.
async fn fetch_pool_rows(
    serials: &[SerialId],
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
) -> Result<(Vec<VectorId>, Vec<ArcIris>, Vec<ArcIris>)> {
    // Source versions are eye-invariant; the LEFT registry is authoritative.
    let maybe_vids = registries[LEFT].get_vector_ids(serials).await;
    let mut vids: Vec<VectorId> = Vec::with_capacity(serials.len());
    for (serial, maybe) in serials.iter().zip(maybe_vids) {
        let vid = maybe.ok_or_else(|| eyre!("row-write serial {serial} missing from registry"))?;
        vids.push(vid);
    }
    let (left_data, right_data) = tokio::try_join!(
        worker_pools[LEFT].fetch_irises(vids.clone()),
        worker_pools[RIGHT].fetch_irises(vids.clone()),
    )?;
    Ok((vids, left_data, right_data))
}

/// Flush the deferred row writes from the worker pools (source content):
/// missing rows INSERTed first, then the update set (replayed serials and
/// tombstone overwrites) rewritten in place. Must run only once the rebuilt
/// graph is durable — see the ordering invariant on [`exec_delta`]. Chunked to
/// bound transaction size.
async fn flush_row_writes(
    repair: &RepairPlan,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: &IrisStore,
) -> Result<()> {
    let (inserts, updates) = repair.row_writes();
    if inserts.is_empty() && updates.is_empty() {
        return Ok(());
    }
    const CHUNK: usize = 1024;

    for chunk in inserts.chunks(CHUNK) {
        let (vids, left_data, right_data) =
            fetch_pool_rows(chunk, registries, worker_pools).await?;
        let refs: Vec<StoredIrisRef> = izip!(&vids, &left_data, &right_data)
            .map(|(vid, left, right)| StoredIrisRef {
                id: vid.serial_id() as i64,
                left_code: &left.code.coefs,
                left_mask: &left.mask.coefs,
                right_code: &right.code.coefs,
                right_mask: &right.mask.coefs,
            })
            .collect();
        let mut tx = hnsw_iris_store.tx().await?;
        hnsw_iris_store
            .insert_copy_irises(&mut tx, &vids, &refs)
            .await?;
        tx.commit().await?;
    }

    for chunk in updates.chunks(CHUNK) {
        let (vids, left_data, right_data) =
            fetch_pool_rows(chunk, registries, worker_pools).await?;
        let mut tx = hnsw_iris_store.tx().await?;
        {
            let mut ev = ExplicitVersionToken::enable(&mut tx).await?;
            for (vid, left, right) in izip!(&vids, &left_data, &right_data) {
                let iris_ref = StoredIrisRef {
                    id: vid.serial_id() as i64,
                    left_code: &left.code.coefs,
                    left_mask: &left.mask.coefs,
                    right_code: &right.code.coefs,
                    right_mask: &right.mask.coefs,
                };
                hnsw_iris_store
                    .update_iris_with_version_id(&mut ev, vid.version_id(), &iris_ref)
                    .await?;
            }
        }
        tx.commit().await?;
    }

    tracing::info!(
        "version-join flushed {} row writes ({} inserts, {} updates)",
        inserts.len() + updates.len(),
        inserts.len(),
        updates.len()
    );
    Ok(())
}

/// Index Iris's from last indexation id.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `s3_client` - AWS S3 client for checkpoint uploads.
/// * `registries` - Per-eye VectorId registries used by the batch generator.
/// * `worker_pools` - Per-eye worker pools that own iris data and cache queries.
/// * `imem_graph_stores` - In-memory graph stores for checkpoints.
/// * `hawk_handle` - Hawk handle managing indexation & search over an HNSW graph.
/// * `tx_results` - Channel to send job results to DB persistence thread.
/// * `task_monitor_bg` - Tokio task monitor to coordinate with process background threads.
/// * `shutdown_handler` - Handler coordinating function termination/process shutdown.
///
#[allow(clippy::too_many_arguments)]
async fn exec_indexation(
    ctx: &ExecutionContextInfo,
    s3_client: &S3Client,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    mut task_monitor_bg: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    tracing::info!(
        "Starting indexation: last_indexed_id={}, max_indexation_id={}",
        ctx.last_indexed_id,
        ctx.args.max_indexation_id
    );

    // Set batch size from config.
    let batch_size = ctx
        .args
        .batch_size_config
        .compute_batch_size(ctx.config.hnsw_param_m);

    if ctx.last_indexed_id + 1 > ctx.args.max_indexation_id {
        tracing::warn!(
            "Last indexed id {} is greater than max indexation id {}. \
                 No indexation will be performed.",
            ctx.last_indexed_id,
            ctx.args.max_indexation_id
        );
    }
    // Set batch generator.
    let mut batch_generator = BatchGenerator::new(
        ctx.last_indexed_id + 1,
        ctx.args.max_indexation_id,
        batch_size,
        ctx.excluded_serial_ids.clone(),
    );
    tracing::info!("Batch generator instantiated: {}", batch_generator);

    // Set indexation result.
    let mut persist_ch: Option<oneshot::Receiver<()>> = None;

    // Checkpoint tracking
    let checkpoint_frequency = ctx.args.checkpoint_frequency;
    // Maximum height at which an intermediate checkpoint would be run, to avoid redundancy with final checkpoint
    let max_intermediate_checkpoint_height = ctx
        .args
        .max_indexation_id
        .saturating_sub(checkpoint_frequency as u32 / 10);
    let mut irises_since_checkpoint: usize = 0;
    let mut last_indexed_id = ctx.last_indexed_id;

    let res: Result<()> = async {
        tracing::info!("Entering main indexation loop");
        tracing::info!(
            "Checkpoint frequency: {} irises per checkpoint",
            checkpoint_frequency
        );

        // Housekeeping.
        let mut now = Instant::now();
        let processing_timeout = Duration::from_secs(ctx.config.processing_timeout_secs);

        // Index until generator is exhausted.
        // N.B. assumes that generator yields non-empty batches containing serial ids > last_indexed_id.
        while let Some(batch) = batch_generator
            .next_batch(last_indexed_id, registries, worker_pools)
            .await?
        {
            // Coordinator: escape on shutdown.
            let shutdown = shutdown_handler.is_shutting_down();
            let mismatch = hawk_handle.sync_state(shutdown, None).await?;
            if shutdown || mismatch {
                tracing::warn!("Shutting down has been triggered");
                break;
            }

            // Coordinator: check background task processing.
            task_monitor_bg.check_tasks();
            last_indexed_id = batch.id_end();
            irises_since_checkpoint += batch.vector_ids.len();

            // Submit batch to Hawk handle for indexation.
            let request = JobRequest::new_batch_indexation(&batch);
            let result_future = hawk_handle.submit_request(request).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    tracing::error!("HawkActor processing timeout: {:?}", err);
                    eyre!("HawkActor processing timeout: {:?}", err)
                })??;

            // Send results to processing thread responsible for persisting to database.
            let (done_rx, result) = result;
            tx_results.send(result).await?;

            // Periodically synchronize batch persistence between nodes.
            let is_sync_batch = (batch.batch_id % PERSIST_DELAY) == PERSIST_DELAY - 1;
            if is_sync_batch {
                if let Some(prev_done_rx) = persist_ch.take() {
                    let wait_start = Instant::now();
                    // Wait for other nodes to finish equivalent persistence.
                    hawk_handle.sync_state(false, Some(prev_done_rx)).await?;
                    metrics::histogram!("genesis_persist_wait_duration")
                        .record(wait_start.elapsed().as_secs_f64());
                }
            }

            // Store current results thread "done" signal channel for future synchronization.
            persist_ch.replace(done_rx);

            metrics::histogram!("genesis_batch_total_duration",
                "synced" => if is_sync_batch { "true" } else { "false" },
            )
            .record(now.elapsed().as_secs_f64());
            tracing::info!(
                "Indexing new batch: {} :: time {:?}s",
                batch,
                now.elapsed().as_secs_f64(),
            );

            // Periodic checkpoint based on snapshot_frequency.  Skipped if close to end of indexation.
            if irises_since_checkpoint >= checkpoint_frequency
                && last_indexed_id <= max_intermediate_checkpoint_height
            {
                upload_and_sync_genesis_checkpoint(
                    &ctx.config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    last_indexed_id,
                    ctx.max_modification_persist_id, // preserve current modification state
                    false, // is_archival: periodic checkpoints are not archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                irises_since_checkpoint = 0;
            };

            now = Instant::now();
        }
        Ok(())
    }
    .await;

    // Process main loop result:
    match res {
        // Success.
        Ok(_) => {
            let wait_start = Instant::now();

            // Create final archival checkpoint if any irises were indexed this run.
            // This runs unconditionally (regardless of periodic checkpoints) to ensure
            // the last checkpoint recorded for a run is always archival.
            if last_indexed_id > ctx.last_indexed_id {
                tracing::info!(
                    "Creating final archival checkpoint: last_indexed_id={}",
                    last_indexed_id
                );
                upload_and_sync_genesis_checkpoint(
                    &ctx.config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    last_indexed_id,
                    ctx.max_modification_persist_id, // preserve current modification state
                    true, // is_archival: final checkpoint is always archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                tracing::info!(
                    "Final archival checkpoint created at iris_id={}",
                    last_indexed_id
                );
            } else if let Some(rx) = persist_ch.take() {
                hawk_handle.sync_state(false, Some(rx)).await?;
            }
            metrics::histogram!("genesis_persist_wait_duration")
                .record(wait_start.elapsed().as_secs_f64());

            hawk_handle.sync_peers().await?;
            tracing::info!("All batches have been processed, shutting down...");

            Ok(())
        }
        Err(err) => {
            tracing::error!("HawkActor processing error: {:?}", err);

            // Clean up & shutdown.
            tracing::info!("Initiating shutdown");
            drop(hawk_handle);
            task_monitor_bg.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor_bg.check_tasks_finished();

            Err(err)
        }
    }
}

/// Takes a dB snapshot.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `aws_rds_client` - AWS RDS SDK client.
///
async fn exec_snapshot(
    ctx: &ExecutionContextInfo,
    aws_rds_client: &RDSClient,
) -> Result<(), IndexationError> {
    tracing::info!("Db snapshot begins");

    // Set snapshot ID.
    let unix_timestamp = Utc::now().timestamp();
    let snapshot_id = format!(
        "genesis-{}-{}-{}-{}",
        ctx.last_indexed_id,
        ctx.args.max_indexation_id,
        ctx.args.batch_size_config.to_aws_identifier(),
        unix_timestamp
    );

    // Set cluster ID.
    let db_config = ctx.config.cpu_database.as_ref().unwrap();
    let url = db_config
        .url
        .strip_prefix("postgresql://")
        .ok_or(IndexationError::AwsRdsInvalidClusterURL)?;
    let at_pos = url
        .rfind('@')
        .ok_or(IndexationError::AwsRdsInvalidClusterURL)?;
    let host_and_db = &url[at_pos + 1..];
    let slash_pos = host_and_db.find('/').unwrap_or(host_and_db.len());
    let cluster_endpoint = &host_and_db[..slash_pos];
    let resp = aws_rds_client
        .describe_db_clusters()
        .send()
        .await
        .map_err(|_| IndexationError::AwsRdsGetClusterURLs)?;
    let cluster_id = resp
        .db_clusters()
        .iter()
        .find(|cluster| cluster.endpoint() == Some(cluster_endpoint))
        .and_then(|cluster| cluster.db_cluster_identifier())
        .ok_or(IndexationError::AwsRdsClusterIdNotFound)?;

    // Create cluster snapshot.
    tracing::info!(
        "Creating RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id,
        snapshot_id.clone()
    );
    aws_rds_client
        .create_db_cluster_snapshot()
        .db_cluster_identifier(cluster_id)
        .db_cluster_snapshot_identifier(snapshot_id.clone())
        .send()
        .await
        .map_err(|err| {
            tracing::error!("Failed to create db snapshot: {}", err);
            IndexationError::AwsRdsCreateSnapshotFailure(err.to_string())
        })?;
    tracing::info!(
        "Created RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id,
        snapshot_id
    );

    Ok(())
}

/// Build `HawkArgs` and the MPC network handle. `init_graph_from_stores`
/// uses them to assemble a fully-loaded `HawkActor`.
async fn build_hawk_networking(
    config: &Config,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<(HawkArgs, Box<dyn iris_mpc_cpu::network::mpc::NetworkHandle>)> {
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;
    let node_addresses: Vec<String> = server_coord_config
        .node_hostnames
        .iter()
        .zip(config.service_ports.iter())
        .map(|(host, port)| format!("{}:{}", host, port))
        .collect();

    let hawk_args = HawkArgs {
        party_index: config.party_id,
        addresses: node_addresses.clone(),
        outbound_addrs: node_addresses.clone(),
        request_parallelism: config.hawk_request_parallelism,
        connection_parallelism: config.hawk_connection_parallelism,
        hnsw_param_ef_constr: config.hnsw_param_ef_constr,
        hnsw_param_m: config.hnsw_param_m,
        hnsw_param_ef_search: config.hnsw_param_ef_search,
        hnsw_param_ef_search_layers_override: config.hnsw_param_ef_search_layers_override.clone(),
        hnsw_param_ef_supermatch: config.hnsw_param_ef_supermatch,
        hnsw_param_ef_saturation_margin: config.hnsw_param_ef_saturation_margin,
        hnsw_layer_density: config.hnsw_layer_density,
        hnsw_min_layer_search_batch_size: config.hnsw_min_layer_search_batch_size,
        hnsw_prf_key: config.hawk_prf_key,
        disable_persistence: config.disable_persistence,
        hnsw_disable_memory_persistence: config.hnsw_disable_memory_persistence,
        tls: config.tls.clone(),
        numa: config.hawk_numa,
    };

    tracing::info!(
        "Initializing Hawk networking (party_index: {}, addresses: {:?})",
        hawk_args.party_index,
        node_addresses
    );

    let networking = build_hawk_network_handle(
        &hawk_args,
        shutdown_handler.get_network_cancellation_token(),
    )
    .await?;
    Ok((hawk_args, networking))
}

/// Returns service clients used downstream.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
async fn get_service_clients(
    config: &Config,
) -> Result<
    (
        (S3Client, S3Client, RDSClient),
        (IrisStore, (IrisStore, GraphPg<Aby3Store<HawkOps>>)),
    ),
    Report,
> {
    /// Returns S3 clients and an RDS client.
    ///
    /// Two S3 clients are constructed so the graph-checkpoint bucket can
    /// live in a different AWS region than the iris-snapshot bucket.
    async fn get_aws_clients(config: &Config) -> Result<(S3Client, S3Client, RDSClient)> {
        let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
        let retry_config = RetryConfig::standard().with_max_attempts(5);

        let config_region = config.aws.clone().and_then(|aws| aws.region);
        let region_name = config_region
            .clone()
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());

        tracing::info!(
            "AWS client init: environment={}, config.aws.region={:?}, \
             effective_region={}, force_path_style={}, max_retry_attempts=5",
            config.environment,
            config_region,
            region_name,
            force_path_style,
        );

        let region = Region::new(region_name.clone());
        let sdk_config = aws_config::from_env().region(region).load().await;

        tracing::info!(
            "Default S3 client: region={}, endpoint={:?}",
            region_name,
            sdk_config.endpoint_url(),
        );

        // S3 client for general AWS operations (iris snapshots, deletions)
        let s3_config = S3ConfigBuilder::from(&sdk_config)
            .force_path_style(force_path_style)
            .retry_config(retry_config.clone())
            .build();
        let aws_s3_client = S3Client::from_conf(s3_config);

        // RDS client using general AWS configuration
        tracing::info!(
            "RDS client: region={}, endpoint={:?}",
            region_name,
            sdk_config.endpoint_url(),
        );
        let rds_client = RDSClient::new(&sdk_config);

        // S3 client for graph checkpoint operations (may be in a different region)
        let checkpoint_region_name = config.graph_checkpoint_bucket_region.clone();
        let checkpoint_region = Region::new(checkpoint_region_name.clone());

        tracing::info!(
            "Checkpoint S3 client: region={}, endpoint={:?}",
            checkpoint_region_name,
            sdk_config.endpoint_url(),
        );

        let checkpoint_s3_config = S3ConfigBuilder::from(&sdk_config)
            .region(checkpoint_region)
            .force_path_style(force_path_style)
            .retry_config(retry_config.clone())
            .build();
        let checkpoint_s3_client = S3Client::from_conf(checkpoint_s3_config);

        Ok((aws_s3_client, checkpoint_s3_client, rds_client))
    }

    /// Returns initialized PostgreSQL clients for both Iris share & HNSW graph stores.
    async fn get_pgres_clients(
        config: &Config,
    ) -> Result<(IrisStore, (IrisStore, GraphPg<Aby3Store<HawkOps>>)), Report> {
        async fn get_mpc_iris_store_client(config: &Config) -> Result<IrisStore, Report> {
            let db_schema = format!(
                "{}{}_{}_{}",
                config.schema_name,
                config.gpu_schema_name_suffix,
                config.environment,
                config.party_id
            );
            let db_config = config
                .database
                .as_ref()
                .ok_or(eyre!("Missing database config"))?;
            tracing::info!(
                "Creating new iris store from: {:?}, schema: {}",
                db_config,
                db_schema
            );
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadOnly)
                    .await?;

            IrisStore::new(&db_client).await
        }

        async fn get_hnsw_store_clients(
            config: &Config,
        ) -> Result<(IrisStore, GraphPg<Aby3Store<HawkOps>>), Report> {
            let db_schema = format!(
                "{}{}_{}_{}",
                config.schema_name,
                config.hnsw_schema_name_suffix,
                config.environment,
                config.party_id
            );
            let db_config = config
                .cpu_database
                .as_ref()
                .ok_or(eyre!("Missing CPU database config for Hawk Genesis"))?;
            tracing::info!(
                "Creating new graph store from: {:?}, schema: {}",
                db_config,
                db_schema
            );
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadWrite)
                    .await?;
            run_migrations(&db_client.pool, db_config.ignore_missing_migrations).await?;

            Ok((
                IrisStore::new(&db_client).await?,
                GraphStore::new(&db_client).await?,
            ))
        }

        Ok((
            get_mpc_iris_store_client(config).await?,
            get_hnsw_store_clients(config).await?,
        ))
    }

    Ok((
        get_aws_clients(config).await?,
        get_pgres_clients(config).await?,
    ))
}

/// Spawns thread responsible for persisting results from batch query processing to database.
///
/// # Arguments
///
/// * `graph_store` - Graph PostgreSQL store provider.
/// * `task_monitor` - Tokio task monitor to coordinate with other threads.
/// * `shutdown_handler` - Handler coordinating process shutdown.
///
async fn get_results_thread(
    worker_pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    hnsw_iris_store: IrisStore,
    graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<Sender<JobResult>> {
    let (tx, mut rx) = mpsc::channel::<JobResult>(PERSIST_DELAY);
    let shutdown_handler_bg = Arc::clone(shutdown_handler);
    let graph_store_bg = Arc::clone(&graph_store);
    let _result_sender_abort = task_monitor.spawn(async move {
        while let Some(result) = rx.recv().await {
            match result {
                // BatchIndexation does not use shutdown_handler to track batches pending completion because it explicitly
                // synchronizes peers instead
                JobResult::BatchIndexation {
                    batch_id,
                    last_serial_id,
                    vector_ids_to_persist,
                    done_tx,
                    ..
                } => {
                    tracing::info!("Job Results :: Received: batch-id={batch_id}");
                    let start = Instant::now();

                    let (left_data, right_data) = tokio::try_join!(
                        worker_pools[LEFT].fetch_irises(vector_ids_to_persist.clone()),
                        worker_pools[RIGHT].fetch_irises(vector_ids_to_persist.clone()),
                    )?;

                    let codes_and_masks: Vec<StoredIrisRef> = vector_ids_to_persist
                            .iter()
                            .enumerate()
                            .map(|(i, vector_id)| {
                                let left_iris = &left_data[i];
                                let right_iris = &right_data[i];
                                StoredIrisRef {
                                    id: vector_id.serial_id() as i64,
                                    left_code: &left_iris.code.coefs,
                                    left_mask: &left_iris.mask.coefs,
                                    right_code: &right_iris.code.coefs,
                                    right_mask: &right_iris.mask.coefs,
                                }
                            })
                            .collect();

                    let mut graph_tx = graph_store.tx().await?;

                    // Persist batch of Iris's to the HNSW graph store.
                    hnsw_iris_store
                        .insert_copy_irises(
                            &mut graph_tx.tx,
                            &vector_ids_to_persist,
                            &codes_and_masks,
                        )
                        .await?;

                    let mut db_tx = graph_tx.tx;
                    set_last_indexed_iris_id(&mut db_tx, last_serial_id).await?;
                    db_tx.commit().await?;
                    tracing::info!(
                        "Job Results :: Persisted last indexed id: batch-id={batch_id}"
                    );

                    tracing::info!(
                        "Job Results :: Persisted to dB: batch-id={batch_id}"
                    );
                    metrics::gauge!("genesis_batch_indexation_complete").set(last_serial_id);
                    metrics::histogram!("genesis_batch_persist_duration").record(start.elapsed().as_secs_f64());
                    let _ = done_tx.send(());
                }
                JobResult::VersionReplay {
                    serial_id,
                    vector_id_to_persist,
                    done_tx,
                } => {
                    tracing::debug!(
                        "Job Results :: Received version replay for serial-id={serial_id}"
                    );
                    // No DB work: the graph persists via the S3 checkpoint and
                    // the iris row via the post-checkpoint flush (a row must
                    // not become source-consistent before the rebuilt graph is
                    // durable). Completion = applied in memory.
                    match vector_id_to_persist {
                        Some(vid) => tracing::info!(
                            "Job Results :: Version replay complete: serial-id={serial_id}, version-id={}",
                            vid.version_id()
                        ),
                        None => tracing::info!(
                            "Job Results :: Version replay complete: serial-id={serial_id}, remove-only"
                        ),
                    }

                    let _ = done_tx.send(());
                    shutdown_handler_bg.decrement_batches_pending_completion();
                }
                JobResult::S3Checkpoint{checkpoint_state, done_tx} => {
                    let graph_tx = graph_store_bg.tx().await?;
                    save_checkpoint_state(graph_tx, &checkpoint_state).await?;
                    let _ = done_tx.send(());
                },
                JobResult::SyncState { .. } | JobResult::SyncPeers => unreachable!(),
            }
        }

        Ok(())
    });

    Ok(tx)
}

/// Build this node's synchronization state, which is compared against the
/// states provided by the other MPC nodes to reconstruct a consistent initial
/// state for MPC operation.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `genesis_config` - Genesis configuration compared for equality across nodes.
///
async fn get_sync_state(
    config: &Config,
    genesis_config: GenesisConfig,
) -> Result<GenesisSyncState> {
    let common_config = CommonConfig::from(config.clone());
    Ok(GenesisSyncState::new(common_config, genesis_config))
}

/// Returns result of performing distributed state synchronization.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `my_state` - Node specific synchronization state information.
///
async fn get_sync_result(
    config: &Config,
    my_state: &GenesisSyncState,
) -> Result<GenesisSyncResult> {
    let mut all_states = vec![my_state.clone()];
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;

    all_states.extend(get_others_sync_state(server_coord_config).await?);
    let result = GenesisSyncResult::new(my_state.clone(), all_states);

    Ok(result)
}

/// Load iris data and the graph (from S3 checkpoint if present, else
/// PG) in parallel, then assemble a fully-loaded `HawkActor`.
/// `max_indexation_id` is the inclusive upper bound for the iris load.
#[allow(clippy::too_many_arguments)]
async fn init_graph_from_stores(
    config: &Config,
    checkpoint_bucket: &str,
    iris_store: &IrisStore,
    hawk_args: HawkArgs,
    hawk_networking: Box<dyn iris_mpc_cpu::network::mpc::NetworkHandle>,
    s3_client: &S3Client,
    shutdown_handler: Arc<ShutdownHandler>,
    max_indexation_id: usize,
    checkpoint: Option<GraphCheckpointState>,
) -> Result<HawkActor> {
    tracing::info!("⚓️ ANCHOR: Load the database");

    let iris_db_parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!(
            "HNSW GENESIS :: Server :: Missing iris database config"
        ))?
        .load_parallelism;
    let graph_db_parallelism = config
        .cpu_database
        .as_ref()
        .ok_or(eyre!(
            "HNSW GENESIS :: Server :: Missing graph database config"
        ))?
        .load_parallelism;
    tracing::info!(
        "Initialize db: Loading from DB with parallelism. iris: {}, graph: {})",
        iris_db_parallelism,
        graph_db_parallelism
    );

    let store_len = iris_store.count_irises().await?;
    let max_index = std::cmp::min(max_indexation_id, store_len);

    let initializer: Box<dyn WorkerPoolInitializer> =
        Box::new(LocalWorkerPoolInitializer::new_load_from_db(
            config.party_id,
            iris_mpc_cpu::execution::hawk_main::HAWK_DISTANCE_MODE,
            config.hawk_numa,
            DbLoadParams {
                store: iris_store.clone(),
                config: Arc::new(config.clone()),
                max_serial_id: max_index,
                parallelism: iris_db_parallelism,
                s3_max_serial_id: Some(max_index),
                shutdown_handler,
            },
        ));

    let graph_load_future = async {
        if let Some(state) = checkpoint {
            tracing::info!(
                "Loading graph from S3 checkpoint, hash: {}",
                state.blake3_hash
            );
            download_graph_checkpoint(s3_client, checkpoint_bucket, &state).await
        } else {
            tracing::info!("No S3 checkpoint found, defaulting to empty graph");
            Ok([GraphMem::new(), GraphMem::new()])
        }
    };

    let (initialized, graph) = tokio::try_join!(initializer.initialize(), graph_load_future)?;

    Ok(HawkActor::new(
        hawk_args,
        hawk_networking,
        initialized,
        graph,
    ))
}

/// Initializes shutdown handler, which waits for shutdown signals or function
/// calls and provides a light mechanism for gracefully finishing ongoing query
/// batches before exiting.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
async fn init_shutdown_handler(config: &Config) -> Arc<ShutdownHandler> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.register_signal_handler().await;

    shutdown_handler
}

/// Validates application config.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
fn validate_config(config: &Config) -> Result<()> {
    // Validate CPU db config.
    if config.cpu_database.is_none() {
        let msg = "Missing CPU dB config settings";
        tracing::error!("{}", msg);
        bail!(msg);
    }

    Ok(())
}

/// Validates consistency of PostGres stores.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `graph_store` - Graph PostgreSQL store provider.
/// * `max_indexation_id` - Maximum Iris serial id to which to index.
/// * `last_indexed_id` - Last Iris serial id to have been indexed.
/// * `checkpoint_available` - Whether an S3 checkpoint is available (skips graph store validation).
///
async fn validate_consistency_of_stores(
    config: &Config,
    iris_store: &IrisStore,
    max_indexation_id: SerialId,
    last_indexed_id: SerialId,
) -> Result<()> {
    // Bail if last indexed id exceeds max indexation id
    if last_indexed_id > max_indexation_id {
        let msg = format!(
            "Last indexed id {} exceeds max indexation id {}",
            last_indexed_id, max_indexation_id
        );
        tracing::error!("{}", msg);
        bail!(msg);
    }

    // Bail if current Iris store length exceeds maximum constraint - should never occur.
    let store_len = iris_store.count_irises().await?;
    if store_len > config.max_db_size {
        let msg = format!(
            "Database size {} exceeds maximum allowed {}",
            store_len, config.max_db_size
        );
        tracing::error!("{}", msg);
        bail!(msg);
    }
    tracing::info!("Size of the database after init: {}", store_len);

    // Bail if max indexation id exceeds max id in the database
    let max_db_id = iris_store.get_max_serial_id().await?;
    if max_indexation_id as usize > max_db_id {
        let msg = format!(
            "Max indexation id {} exceeds max database id {}",
            max_indexation_id, max_db_id
        );
        tracing::error!("{}", msg);
        bail!(msg);
    }
    Ok(())
}
