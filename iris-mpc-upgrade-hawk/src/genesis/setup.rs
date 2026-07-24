//! Process setup for genesis indexing.
//!
//! Brings the node up to the point where indexation can run: service clients,
//! checkpoint negotiation with the peers over the coordination server,
//! networking sync and rollback to the common checkpoint, and the parallel
//! iris/graph load that assembles the `HawkActor`. The public entry point is
//! [`exec_setup`], which returns a [`SetupOutput`] consumed by the phases in
//! the parent module.

use ampc_server_utils::{
    get_others_sync_state, init_heartbeat_task, set_node_ready, shutdown_handler::ShutdownHandler,
    start_coordination_server_with_extra_routes, wait_for_others_ready, wait_for_others_unready,
    BatchSyncSharedState, TaskMonitor,
};
use aws_sdk_rds::Client as RDSClient;
use aws_sdk_s3::{config::Region, Client as S3Client};
use axum::{routing::get, Router};
use eyre::{bail, eyre, Report, Result};
use futures::TryStreamExt;
use iris_mpc_common::{
    config::{CommonConfig, Config, ENV_PROD, ENV_STAGE},
    postgres::{run_migrations, AccessMode, PostgresClient},
    SerialId,
};
use iris_mpc_cpu::{
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
            set_last_indexed_iris_id,
        },
        state_sync::{
            Config as GenesisConfig, SyncResult as GenesisSyncResult, SyncState as GenesisSyncState,
        },
        Handle as GenesisHawkHandle, JobResult,
    },
    graph_checkpoint::*,
    hawkers::aby3::aby3_store::{Aby3Store, VectorIdRegistryRef},
    hnsw::{graph::graph_store::GraphPg, GraphMem},
    utils::serialization::graph::{GraphFormat, LegacyPruneContext, PruneReport},
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, Sender};

use super::delta::{
    delta_slot_route, DeltaExchangeSlots, DELTA_REPAIR_ROUTE, DELTA_TOMBSTONES_ROUTE,
};
use super::graph_checkpoint::reset_to_checkpoint;
use super::retry::{with_retry, DB_RETRY_ATTEMPTS};
use super::{ExecutionArgs, ExecutionContextInfo, PERSIST_DELAY};

const DEFAULT_REGION: &str = "eu-north-1";

/// Everything [`exec_setup`] hands back to the parent `exec` orchestrator: the
/// execution context plus the long-lived clients, handles and channels the
/// later phases operate over.
pub(super) struct SetupOutput {
    pub(super) ctx: ExecutionContextInfo,
    pub(super) shutdown_handler: Arc<ShutdownHandler>,
    pub(super) task_monitor_bg: TaskMonitor,
    pub(super) checkpoint_s3_client: S3Client,
    pub(super) aws_rds_client: RDSClient,
    pub(super) registries: BothEyes<VectorIdRegistryRef>,
    pub(super) worker_pools: BothEyes<Arc<dyn IrisWorkerPool>>,
    pub(super) imem_graph_stores: Arc<BothEyes<GraphRef>>,
    pub(super) hawk_handle: GenesisHawkHandle,
    pub(super) tx_results: Sender<JobResult>,
    pub(super) graph_store: Arc<GraphPg<Aby3Store<HawkOps>>>,
    pub(super) hnsw_iris_store: IrisStore,
    pub(super) delta_exchange: DeltaExchangeSlots,
    /// Per-eye damage census from pruning a legacy (V3/V4) base at load;
    /// `None` on a native V5 base or fresh start.
    pub(super) prune_reports: Option<[PruneReport; 2]>,
}

/// Execute process setup tasks.
///
/// # Arguments
///
/// * `args` - Process arguments.
/// * `config` - Process configuration instance.
///
pub(super) async fn exec_setup(args: &ExecutionArgs, config: &Config) -> Result<SetupOutput> {
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
    let (hawk_actor, prune_reports) = init_graph_from_stores(
        config,
        &config.graph_checkpoint_bucket_name,
        &iris_store,
        hawk_args,
        hawk_networking,
        &checkpoint_s3_client,
        Arc::clone(&shutdown_handler),
        args.max_indexation_id as usize,
        graph_checkpoint.clone(),
        excluded_serial_ids.iter().copied().collect(),
    )
    .await?;
    task_monitor_bg.check_tasks();
    tracing::info!("HNSW graph initialised from store");
    if let Some(reports) = prune_reports.as_ref() {
        log_prune_reports(reports);
    }

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
    let hawk_handle = GenesisHawkHandle::new(
        hawk_actor,
        Duration::from_secs(config.genesis_sync_timeout_secs),
    )
    .await?;
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

    Ok(SetupOutput {
        ctx: ExecutionContextInfo::new(
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
        graph_store: graph_store_arc,
        hnsw_iris_store,
        delta_exchange,
        prune_reports,
    })
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
        let aws_s3_client = create_s3_client(&sdk_config, force_path_style, None);

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

        let checkpoint_s3_client =
            create_s3_client(&sdk_config, force_path_style, Some(checkpoint_region));

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
    deleted_serial_ids: HashSet<SerialId>,
) -> Result<(HawkActor, Option<[PruneReport; 2]>)> {
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

    let prune_iris_store = iris_store.clone();
    let graph_load_future = async move {
        if let Some(state) = checkpoint {
            tracing::info!(
                "Loading graph from S3 checkpoint, hash: {}",
                state.blake3_hash
            );
            // Legacy V3/V4 bases are pruned at read (see `read_graph_pair_pruned`);
            // only they need the current-version table, so skip the scan for V5.
            let prune = match GraphFormat::try_from(state.graph_version)? {
                format @ (GraphFormat::V3 | GraphFormat::V4) => {
                    let version_map: HashMap<u32, i16> =
                        with_retry("version_map scan", DB_RETRY_ATTEMPTS, || {
                            let store = &prune_iris_store;
                            async move {
                                store
                                    .stream_iris_ids(max_index)
                                    .map_ok(|(id, version)| {
                                        (u32::try_from(id).expect("serial id fits u32"), version)
                                    })
                                    .try_collect()
                                    .await
                                    .map_err(eyre::Report::from)
                            }
                        })
                        .await?;
                    // The prune classifies any serial absent from `version_map`
                    // as stale and silently drops it. The scan bound `max_index`
                    // (= min(max_indexation_id, count)) must reach the base
                    // height; otherwise the uncovered tail past it is prune-
                    // erased — live serials the join can never repair, since both
                    // pools are bounded identically. Within-bound holes prune
                    // silently, which is acceptable: a graph node without source
                    // content is unusable regardless.
                    let base_height = state.last_indexed_iris_id;
                    if max_index < base_height as usize {
                        bail!(
                            "version_map scan bound ({max_index}) does not reach the {format} \
                             base height ({base_height}); the uncovered tail would be prune-erased"
                        );
                    }
                    // Downstream, the version-join sees pruned serials only as
                    // absent from the graph — the specific drop cause is
                    // consumed here and reported via the prune report.
                    tracing::info!(
                        "Legacy {format} base: prune-at-read precedes the \
                         version-join; join reasons for pruned serials degrade \
                         to graph-missing"
                    );
                    Some(LegacyPruneContext {
                        version_map,
                        deleted: deleted_serial_ids,
                    })
                }
                _ => None,
            };
            download_graph_checkpoint_pruned(s3_client, checkpoint_bucket, &state, prune).await
        } else {
            tracing::info!("No S3 checkpoint found, defaulting to empty graph");
            Ok(([GraphMem::new(), GraphMem::new()], None))
        }
    };

    let (initialized, (graph, prune_reports)) =
        tokio::try_join!(initializer.initialize(), graph_load_future)?;

    Ok((
        HawkActor::new(hawk_args, hawk_networking, initialized, graph),
        prune_reports,
    ))
}

/// Log the per-eye damage census from a legacy-base prune: class counts, then
/// serial lists in bounded chunks (CloudWatch truncates oversized lines).
fn log_prune_reports(reports: &[PruneReport; 2]) {
    const CHUNK: usize = 1000;
    let log_serials = |eye: &str, class: &str, serials: Vec<u32>| {
        for (i, chunk) in serials.chunks(CHUNK).enumerate() {
            tracing::info!(
                "Prune report [{eye}] {class} serials ({}/{}): {chunk:?}",
                i * CHUNK + chunk.len(),
                serials.len(),
            );
        }
    };
    for (report, eye) in reports.iter().zip(["left", "right"]) {
        tracing::info!(
            "Prune report [{eye}]: multi_version={} self_loop={} \
             nodes_dropped(deleted={} stale={}) \
             edges_dropped(deleted={} stale={} self_loop={}) \
             zero_out_degree={} zero_in_degree={}",
            report.multi_version_serials.len(),
            report.self_loop_serials.len(),
            report.nodes_dropped_deleted,
            report.nodes_dropped_stale,
            report.edges_dropped_deleted,
            report.edges_dropped_stale,
            report.edges_dropped_self_loop,
            report.zero_out_degree.len(),
            report.zero_in_degree.len(),
        );
        log_serials(
            eye,
            "multi_version",
            report.multi_version_serials.iter().copied().collect(),
        );
        log_serials(
            eye,
            "self_loop",
            report.self_loop_serials.iter().copied().collect(),
        );
        log_serials(eye, "zero_out_degree", report.zero_out_degree.clone());
        log_serials(eye, "zero_in_degree", report.zero_in_degree.clone());
    }
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
