use ampc_server_utils::{
    get_others_sync_state, init_heartbeat_task, set_node_ready, shutdown_handler::ShutdownHandler,
    start_coordination_server, wait_for_others_ready, wait_for_others_unready,
    BatchSyncSharedState, TaskMonitor,
};
use aws_config::retry::RetryConfig;
use aws_sdk_rds::Client as RDSClient;
use aws_sdk_s3::{
    config::{Builder as S3ConfigBuilder, Region},
    Client as S3Client,
};
use chrono::Utc;
use eyre::{bail, eyre, Report, Result};

use iris_mpc_common::{
    config::{CommonConfig, Config, ENV_PROD, ENV_STAGE},
    helpers::{smpc_request, sync::Modification},
    postgres::{AccessMode, PostgresClient},
    IrisSerialId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, GraphStore, HawkActor, HawkArgs, StoreId, LEFT, RIGHT},
    genesis::{
        state_accessor::{
            get_iris_deletions, get_iris_modifications, get_last_indexed_iris_id,
            get_last_indexed_modification_id, set_last_indexed_iris_id,
            set_last_indexed_modification_id,
        },
        state_sync::{
            Config as GenesisConfig, SyncResult as GenesisSyncResult, SyncState as GenesisSyncState,
        },
        utils, BatchGenerator, BatchIterator, BatchSize, Handle as GenesisHawkHandle,
        IndexationError, JobRequest, JobResult,
    },
    hawkers::aby3::aby3_store::{Aby3SharedIrisesRef, Aby3Store},
    hnsw::graph::graph_store::GraphPg,
};
use iris_mpc_store::{loader::load_iris_db, Store as IrisStore, StoredIrisRef};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc::{self, Sender},
    time::timeout,
};

const DEFAULT_REGION: &str = "eu-north-1";

/// Process input arguments typically passed from command line.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionArgs {
    // Serial idenitifer of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Initial batch size for indexing.
    batch_size: usize,

    // Error rate to be applied when calculating dynamic batch sizes.
    batch_size_error_rate: usize,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    perform_snapshot: bool,

    // Use backup as source
    use_backup_as_source: bool,
}

/// Constructor.
impl ExecutionArgs {
    pub fn new(
        batch_size: usize,
        batch_size_error_rate: usize,
        max_indexation_id: IrisSerialId,
        perform_snapshot: bool,
        use_backup_as_source: bool,
    ) -> Self {
        Self {
            batch_size,
            batch_size_error_rate,
            max_indexation_id,
            perform_snapshot,
            use_backup_as_source,
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
    last_indexed_id: IrisSerialId,

    // Set identifiers of Iris's to be excluded from indexation.
    excluded_serial_ids: Vec<IrisSerialId>,

    // Set of modifications to be applied.
    modifications: Vec<Modification>,

    // Maximum modification id to be performed
    max_modification_indexed_id: i64,

    // The largest modification id that has been completed and persisted by the source version.
    // Used to track up to which modification the next run of Genesis can start from
    max_modification_persist_id: i64,
}

/// Constructor.
impl ExecutionContextInfo {
    fn new(
        args: &ExecutionArgs,
        config: &Config,
        last_indexed_id: IrisSerialId,
        excluded_serial_ids: Vec<IrisSerialId>,
        modifications: Vec<Modification>,
        max_modification_indexed_id: i64,
        max_modification_persist_id: i64,
    ) -> Self {
        Self {
            args: *args,
            config: config.clone(),
            excluded_serial_ids,
            last_indexed_id,
            modifications,
            max_modification_indexed_id,
            max_modification_persist_id,
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
    // Phase 0: setup.
    let (
        ctx,
        shutdown_handler,
        mut task_monitor_bg,
        aws_rds_client,
        imem_iris_stores,
        mut hawk_handle,
        tx_results,
        graph_store,
        hnsw_iris_store,
    ) = exec_setup(&args, &config).await?;

    log_info(String::from("Setup complete."));
    log_info(format!(
        "Starting Genesis indexing process with the following parameters:\n  Max indexation ID: {}\n  Batch size: {}\n  Batch size error rate: {}\n  Perform snapshot: {}\n  User backup as source: {}\n  Persistence enabled: {}",
        args.max_indexation_id,
        args.batch_size,
        args.batch_size_error_rate,
        args.perform_snapshot,
        args.use_backup_as_source,
        !config.disable_persistence,
    ));

    // Phase 1: apply delta.
    hawk_handle = exec_delta(
        &config,
        &ctx,
        graph_store.clone(),
        hawk_handle,
        &tx_results,
        &mut task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    log_info(String::from("Delta complete."));

    // Phase 2: indexation.
    exec_indexation(
        &ctx,
        &imem_iris_stores,
        hawk_handle,
        &tx_results,
        task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    log_info(String::from("Indexation complete."));

    // Phase 3: database backup.
    if !config.disable_persistence {
        log_info(String::from("Database backup begins"));
        exec_database_backup(graph_store.clone()).await?;
    } else {
        log_info(String::from(
            "Database backup skipped (persistence disabled)",
        ));
    }

    // Phase 4: snapshot.
    if !args.perform_snapshot {
        log_info(String::from("Snapshot skipped ... as requested."));
    } else {
        exec_snapshot(&ctx, &aws_rds_client).await?;
        log_info(String::from("Snapshot complete."));
    };

    // Clear modifications from the HNSW iris store
    // This is because after a genesis run - there should be no modifications left in the HNSW iris store
    if !config.disable_persistence {
        let mut tx = hnsw_iris_store.tx().await?;
        hnsw_iris_store
            .clear_modifications_table(&mut tx)
            .await
            .map_err(|err| {
                eyre!(log_error(format!(
                    "Failed to clear modifications: {:?}",
                    err
                )))
            })?;
        tx.commit().await?;

        log_info(String::from(
            "Cleared modifications from the HNSW iris store",
        ));
    } else {
        log_info(String::from(
            "Persistence disabled, skipping modifications table cleanup",
        ));
    }

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
    RDSClient,
    Arc<BothEyes<Aby3SharedIrisesRef>>,
    GenesisHawkHandle,
    Sender<JobResult>,
    Arc<GraphPg<Aby3Store>>,
    IrisStore,
)> {
    // Bail if config is invalid.
    validate_config(config)?;

    // Set shutdown handler.
    let shutdown_handler = init_shutdown_handler(config).await;

    // Set background task monitor.
    let mut task_monitor_bg = TaskMonitor::new();

    // Set service clients.
    let ((aws_s3_client, aws_rds_client), (iris_store, (hnsw_iris_store, graph_store))) =
        get_service_clients(config).await?;
    log_info(String::from("Service clients instantiated"));
    let graph_store_arc = Arc::new(graph_store);

    // Set serial identifier of last indexed Iris.
    let last_indexed_id = get_last_indexed_iris_id(graph_store_arc.clone()).await?;
    log_info(format!(
        "Identifier of last Iris to have been indexed = {}",
        last_indexed_id,
    ));

    // Set Iris serial identifiers marked for deletion and thus excluded from indexation.
    let excluded_serial_ids =
        get_iris_deletions(config, &aws_s3_client, args.max_indexation_id).await?;
    log_info(format!(
        "Deletions for exclusion count = {}",
        excluded_serial_ids.len(),
    ));

    // Set modifications that have occurred since last indexation.
    let last_indexed_modification_id =
        get_last_indexed_modification_id(graph_store_arc.clone()).await?;
    log_info(format!(
        "Identifier of last modification to have been indexed = {}",
        last_indexed_modification_id,
    ));
    // This is the largest modification id that has been completed by the node.
    let (modifications, max_modification_id_to_persist) =
        get_iris_modifications(&iris_store, last_indexed_modification_id, last_indexed_id).await?;
    let max_modification_id = modifications.last().map_or(0, |m| m.id);
    log_info(format!(
        "Modifications to be applied count = {}. Max modification id completed = {}, Max modification to be performed = {}",
        modifications.len(),
        max_modification_id_to_persist,
        max_modification_id,
    ));

    // Coordinator: Await coordination server to start.
    let genesis_config = GenesisConfig::new(
        args.batch_size,
        args.batch_size_error_rate,
        excluded_serial_ids.clone(),
        last_indexed_id,
        args.max_indexation_id,
        max_modification_id,
        max_modification_id_to_persist,
        modifications.clone(),
        args.use_backup_as_source,
    );
    let my_state = get_sync_state(config, genesis_config).await?;
    log_info(String::from("Synchronization state initialised"));

    let batch_sync_shared_state =
        Arc::new(tokio::sync::Mutex::new(BatchSyncSharedState::default()));

    let server_coord_config = &config
        .server_coordination
        .clone()
        .unwrap_or_else(|| panic!("Server coordination config is required for server operation"));
    // Coordinator: await server start.
    let (is_ready_flag, verified_peers, my_uuid) = start_coordination_server(
        server_coord_config,
        &mut task_monitor_bg,
        &shutdown_handler,
        &my_state,
        Some(batch_sync_shared_state),
    )
    .await;
    task_monitor_bg.check_tasks();

    let server_coord_config = &config
        .server_coordination
        .clone()
        .unwrap_or_else(|| panic!("Server coordination config is required for server operation"));
    // Coordinator: await network state = UNREADY.
    wait_for_others_unready(server_coord_config, &verified_peers, &my_uuid).await?;
    log_info(String::from("Network status = UNREADY"));
    // Coordinator: await network state = HEALTHY.
    init_heartbeat_task(server_coord_config, &mut task_monitor_bg, &shutdown_handler).await?;
    task_monitor_bg.check_tasks();
    log_info(String::from("Network status = HEALTHY"));

    // Coordinator: await network state = SYNCHRONIZED.
    let sync_result = get_sync_result(config, &my_state).await?;
    sync_result.check_synced_state()?;
    log_info(String::from("Synchronization checks passed"));

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        log_warn(String::from("Shutting down has been triggered"));
        bail!("Shutdown")
    }

    // If use_backup_as_source is set, restore graph tables from backup
    if args.use_backup_as_source && last_indexed_id != 0 {
        if config.disable_persistence {
            log_warn(String::from(
                "Persistence is disabled but use_backup_as_source is set - proceeding with backup restore"
            ));
        }
        exec_use_backup_as_source(
            last_indexed_id,
            &graph_store_arc,
            &hnsw_iris_store,
            &iris_store,
        )
        .await?;
    }

    // Bail if stores are inconsistent.
    validate_consistency_of_stores(
        config,
        &iris_store,
        graph_store_arc.clone(),
        args.max_indexation_id,
        last_indexed_id,
    )
    .await?;
    log_info(String::from("Store consistency checks OK"));

    // Initialise HNSW graph from previously indexed.
    let mut hawk_actor = get_hawk_actor(config, &shutdown_handler).await?;
    hawk_actor.sync_peers().await?;
    init_graph_from_stores(
        config,
        &iris_store,
        graph_store_arc.clone(),
        &mut hawk_actor,
        Arc::clone(&shutdown_handler),
        args.max_indexation_id as usize,
    )
    .await?;
    task_monitor_bg.check_tasks();
    log_info(String::from("HNSW graph initialised from store"));

    // Coordinator: await network state = ready.
    set_node_ready(is_ready_flag);
    let ct = shutdown_handler.get_network_cancellation_token();
    tokio::select! {
        _ = ct.cancelled() => Err(eyre!("ready check failed")),
        r = wait_for_others_ready(server_coord_config) => r
    }?;
    task_monitor_bg.check_tasks();
    log_info(String::from("Network status = READY"));

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        log_warn(String::from("Shutting down has been triggered"));
        bail!("Shutdown")
        // return Ok(());
    }

    // Set in memory Iris stores.
    let imem_iris_stores = Arc::new([
        hawk_actor.iris_store(StoreId::Left),
        hawk_actor.iris_store(StoreId::Right),
    ]);

    // Set Hawk handle.
    let hawk_handle = GenesisHawkHandle::new(hawk_actor).await?;
    log_info(String::from("Hawk handle initialised"));

    // Set thread for persisting indexing results to DB.
    let tx_results = get_results_thread(
        Arc::clone(&imem_iris_stores),
        hnsw_iris_store.clone(),
        graph_store_arc.clone(),
        &mut task_monitor_bg,
        &shutdown_handler,
        config.disable_persistence,
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
            max_modification_id,
            max_modification_id_to_persist,
        ),
        shutdown_handler,
        task_monitor_bg,
        aws_rds_client,
        imem_iris_stores,
        hawk_handle,
        tx_results,
        graph_store_arc,
        hnsw_iris_store,
    ))
}

/// Apply modifications since last indexation.
///
/// # Arguments
///
/// * `config` - Application configuration struct.
/// * `ctx` - Execution context information.
/// * `hawk_handle` - Genesis hawk handle for processing queries with HNSW engine.
/// * `tx_results` - Sender handle for persisting modifications to database.
/// * `task_monitor_bg` - Tokio task monitor to coordinate with process background threads.
/// * `shutdown_handler` - Handler coordinating function termination/process shutdown.
///
async fn exec_delta(
    config: &Config,
    ctx: &ExecutionContextInfo,
    graph_store: Arc<GraphPg<Aby3Store>>,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    task_monitor_bg: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<GenesisHawkHandle> {
    let ExecutionContextInfo {
        modifications,
        max_modification_indexed_id,
        max_modification_persist_id,
        ..
    } = ctx;

    let res: Result<()> = async {
        if modifications.is_empty() {
            log_info(String::from("Delta has no modifications to apply."));
            return Ok(());
        }
        log_info(format!(
            "Applying modifications: count={} :: max-id={}",
            modifications.len(),
            max_modification_indexed_id
        ));

        metrics::gauge!("genesis_number_modifications").set(modifications.len() as f64);
        metrics::gauge!("genesis_max_modification_id").set(*max_modification_indexed_id as f64);

        let processing_timeout = Duration::from_secs(config.processing_timeout_secs);

        for modification in modifications {
            log_info(format!(
                "Applying modification: type={} id={}, serial_id={:?}",
                modification.request_type, modification.id, modification.serial_id
            ));
            if modification.request_type == smpc_request::IDENTITY_DELETION_MESSAGE_TYPE {
                if ctx.config.environment != ENV_PROD {
                    log_info(format!(
                        "Modification is a deletion: serial_id={:?} and it is not production therefore skipping",
                        modification.serial_id
                    ));
                    continue;
                }else {
                    bail!(eyre!(
                        "Modification is a deletion: serial_id={:?} and it is production therefore bailing",
                        modification.serial_id
                    ));
                }
            }

            // Submit modification to Hawk handle for processing.
            let request = JobRequest::new_modification(modification.clone());
            let result_future = hawk_handle.submit_request(request).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    eyre!(log_error(format!(
                        "HawkActor processing timeout: {:?}",
                        err
                    )))
                })??;

            // Send results to processing thread responsible for persisting to database.
            let (done_rx, result) = result;
            tx_results.send(result).await?;
            shutdown_handler.increment_batches_pending_completion();
            done_rx.await?;
            hawk_handle.sync_peers().await?;
        }

        Ok(())
    }
    .await;

    // Process delta result:
    match res {
        // Success.
        Ok(_) => {
            log_info(String::from(
                "Waiting for last delta modifications to be processed...",
            ));
            shutdown_handler.wait_for_pending_batches_completion().await;
            log_info(String::from("All delta modifications have been processed"));

            if !config.disable_persistence {
                log_info(format!( "Setting last indexed modification id to the largest completed and persisted modification id = {}", max_modification_persist_id));
                let mut graph_tx = graph_store.tx().await?;
                set_last_indexed_modification_id(&mut graph_tx.tx, *max_modification_persist_id)
                    .await?;
                graph_tx.tx.commit().await?;
            } else {
                log_info(
                    "Persistence disabled, skipping last indexed modification ID update"
                        .to_string(),
                );
            }

            Ok(hawk_handle)
        }
        // Error.
        Err(err) => {
            log_error(format!(
                "HawkActor processing error while applying delta modifications: {:?}",
                err
            ));

            // Clean up & shutdown.
            log_info(String::from("Initiating shutdown"));
            drop(hawk_handle);
            task_monitor_bg.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor_bg.check_tasks_finished();

            Err(err)
        }
    }
}

/// Index Iris's from last indexation id.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `imem_iris_stores` - In-memory iris shares for indexation queries.
/// * `hawk_actor` - Hawk actor managing indexation & search over an HNSW graph.
/// * `tx_results` - Channel to send job results to DB persistence thread.
/// * `task_monitor_bg` - Tokio task monitor to coordinate with process background threads.
/// * `shutdown_handler` - Handler coordinating function termination/process shutdown.
///
async fn exec_indexation(
    ctx: &ExecutionContextInfo,
    imem_iris_stores: &BothEyes<Aby3SharedIrisesRef>,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    mut task_monitor_bg: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    log_info(format!(
        "Starting indexation: last_indexed_id={}, max_indexation_id={}",
        ctx.last_indexed_id, ctx.args.max_indexation_id
    ));

    // Set batch size.
    let batch_size = match ctx.args.batch_size {
        0 => BatchSize::new_dynamic(ctx.args.batch_size_error_rate, ctx.config.hnsw_param_M),
        _ => BatchSize::new_static(ctx.args.batch_size),
    };

    if ctx.last_indexed_id + 1 > ctx.args.max_indexation_id {
        log_warn(format!(
            "Last indexed id {} is greater than max indexation id {}. \
                 No indexation will be performed.",
            ctx.last_indexed_id, ctx.args.max_indexation_id
        ));
    }
    // Set batch generator.
    let mut batch_generator = BatchGenerator::new(
        ctx.last_indexed_id + 1,
        ctx.args.max_indexation_id,
        batch_size,
        ctx.excluded_serial_ids.clone(),
    );
    log_info(format!("Batch generator instantiated: {}", batch_generator));

    // Set indexation result.
    let res: Result<()> = async {
        log_info(String::from("Entering main indexation loop"));

        // Housekeeping.
        let mut now = Instant::now();
        let processing_timeout = Duration::from_secs(ctx.config.processing_timeout_secs);

        // Index until generator is exhausted.
        // N.B. assumes that generator yields non-empty batches containing serial ids > last_indexed_id.
        let mut last_indexed_id = ctx.last_indexed_id;
        while let Some(batch) = batch_generator
            .next_batch(last_indexed_id, imem_iris_stores)
            .await?
        {
            // Coordinator: escape on shutdown.
            if shutdown_handler.is_shutting_down() {
                log_warn(String::from("Shutting down has been triggered"));
                break;
            }

            // Coordinator: check background task processing.
            task_monitor_bg.check_tasks();

            last_indexed_id = batch.id_end();

            // Submit batch to Hawk handle for indexation.
            let request = JobRequest::new_batch_indexation(&batch);
            let result_future = hawk_handle.submit_request(request).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    eyre!(log_error(format!(
                        "HawkActor processing timeout: {:?}",
                        err
                    )))
                })??;

            // Send results to processing thread responsible for persisting to database.
            let (done_rx, result) = result;
            tx_results.send(result).await?;
            shutdown_handler.increment_batches_pending_completion();
            // Signal.
            log_info(format!(
                "Indexing new batch: {} :: time {:?}s",
                batch,
                now.elapsed().as_secs_f64(),
            ));
            done_rx.await?;
            hawk_handle.sync_peers().await?;
            now = Instant::now();
        }
        Ok(())
    }
    .await;

    // Process main loop result:
    match res {
        // Success.
        Ok(_) => {
            log_info(String::from(
                "Waiting for last batch results to be processed before \
                 shutting down...",
            ));
            shutdown_handler.wait_for_pending_batches_completion().await;
            log_info(String::from(
                "All batches have been processed, \
                 shutting down...",
            ));

            Ok(())
        }
        // Error.
        Err(err) => {
            log_error(format!("HawkActor processing error: {:?}", err));

            // Clean up & shutdown.
            log_info(String::from("Initiating shutdown"));
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
    log_info(String::from("Db snapshot begins"));

    // Set snapshot ID.
    let unix_timestamp = Utc::now().timestamp();
    let snapshot_id = format!(
        "genesis-{}-{}-{}-{}",
        ctx.last_indexed_id, ctx.args.max_indexation_id, ctx.args.batch_size, unix_timestamp
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
    log_info(format!(
        "Creating RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id,
        snapshot_id.clone()
    ));
    aws_rds_client
        .create_db_cluster_snapshot()
        .db_cluster_identifier(cluster_id)
        .db_cluster_snapshot_identifier(snapshot_id.clone())
        .send()
        .await
        .map_err(|err| {
            log_error(format!("Failed to create db snapshot: {}", err));
            IndexationError::AwsRdsCreateSnapshotFailure(err.to_string())
        })?;
    log_info(format!(
        "Created RDS snapshot for cluster: cluster-id={} :: snapshot-id={}",
        cluster_id, snapshot_id
    ));

    Ok(())
}

/// Executes database backup by copying schema and table data.
///
/// # Arguments
///
/// * `graph_store` - Arc-wrapped HNSW graph store instance.
async fn exec_database_backup(graph_store: Arc<GraphPg<Aby3Store>>) -> Result<(), IndexationError> {
    log_info(String::from("Graph table data snapshot begins"));
    let now = Instant::now();
    graph_store
        .backup_hawk_graph_tables()
        .await
        .map_err(|err| {
            log_error(format!("Failed to copy table data: {}", err));
            IndexationError::DatabaseCopyFailure(err.to_string())
        })?;
    log_info(format!(
        "Graph table data snapshot ended - time taken is {:?}s",
        now.elapsed().as_secs_f64()
    ));

    Ok(())
}

/// Restores the HNSW graph and iris data from backup if `use_backup_as_source` is set.
///
/// This method is used when HNSW persistence is enabled and the graph/iris data may have been modified.
/// It restores the HNSW schema and data to the last known consistent state by:
///   1. Restoring graph tables from backup.
///   2. Removing all iris data with serial IDs greater than the last indexed ID.
///   3. Overriding iris data in the HNSW iris store for all modifications recorded in the modifications table,
///      by copying the corresponding iris data from the main iris store.
///
/// # Arguments
///
/// * `last_indexed_id` - The last indexed iris serial ID to keep in the HNSW iris store.
/// * `graph_store_arc` - Arc-wrapped HNSW graph store instance.
/// * `hnsw_iris_store` - The HNSW iris store to restore.
/// * `iris_store` - The main iris store to copy data from.
pub async fn exec_use_backup_as_source(
    last_indexed_id: u32,
    graph_store_arc: &Arc<GraphPg<Aby3Store>>,
    hnsw_iris_store: &IrisStore,
    iris_store: &IrisStore,
) -> Result<()> {
    log_info(String::from(
        "Restoring graph tables from backup as user_backup_as_source is set",
    ));

    // Step 1: Restore graph tables from backup.
    let mut now = Instant::now();
    graph_store_arc
        .restore_hawk_graph_tables_from_backup()
        .await?;
    log_info(format!(
        "Graph tables restored from backup :: time {:?}s",
        now.elapsed().as_secs_f64()
    ));

    // Step 2: Remove all iris data except that is larger than the last indexed id.
    now = Instant::now();
    hnsw_iris_store.rollback(last_indexed_id as usize).await?;
    log_info(format!(
        "Removing all iris data except that larger than last indexed id: {}:: time {:?}s",
        last_indexed_id,
        now.elapsed().as_secs_f64()
    ));

    // Step 3: Use modifications table created during the last Genesis run to override the iris data in the HNSW iris store.
    // In the case that HNSW performed some modification that GPU did not, we would need to override the iris data
    now = Instant::now();
    let max_hnsw_serial_id = hnsw_iris_store.get_max_serial_id().await?;
    let (hnsw_mods, _max_id) = hnsw_iris_store
        .get_persisted_modifications_after_id(0, max_hnsw_serial_id as u32)
        .await?;
    if !hnsw_mods.is_empty() {
        log_info(format!("Restoring {} iris modifications", hnsw_mods.len()));
        let mut tx = hnsw_iris_store.tx().await?;
        for modification in &hnsw_mods {
            if let Some(serial_id) = modification.serial_id {
                log_info(format!(
                    "Restoring iris modification: id={}, serial_id={:?}",
                    modification.id, modification.serial_id
                ));

                let iris = iris_store.get_iris_data_by_id(serial_id).await?;
                let iris_ref = iris_mpc_store::StoredIrisRef {
                    id: iris.serial_id() as i64,
                    left_code: iris.left_code(),
                    left_mask: iris.left_mask(),
                    right_code: iris.right_code(),
                    right_mask: iris.right_mask(),
                };
                hnsw_iris_store
                    .update_iris_with_version_id(Some(&mut tx), iris.version_id(), &iris_ref)
                    .await?;
            }
        }
        tx.commit().await?;
    }
    log_info(format!(
        "Restoring iris data from modifications table in HNSW iris store :: time {:?}s",
        now.elapsed().as_secs_f64()
    ));

    Ok(())
}

/// Factory function to return a configured Hawk actor that manages HNSW graph construction & search.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
async fn get_hawk_actor(
    config: &Config,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<HawkActor> {
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
        hnsw_param_M: config.hnsw_param_M,
        hnsw_param_ef_search: config.hnsw_param_ef_search,
        hnsw_prf_key: config.hawk_prf_key,
        disable_persistence: config.disable_persistence,
        match_distances_buffer_size: config.match_distances_buffer_size,
        n_buckets: config.n_buckets,
        tls: config.tls.clone(),
        numa: config.hawk_numa,
    };

    log_info(format!(
        "Initializing HawkActor with args: party_index: {}, addresses: {:?}",
        hawk_args.party_index, node_addresses
    ));

    HawkActor::from_cli(
        &hawk_args,
        shutdown_handler.get_network_cancellation_token(),
    )
    .await
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
        (S3Client, RDSClient),
        (IrisStore, (IrisStore, GraphPg<Aby3Store>)),
    ),
    Report,
> {
    /// Returns an S3 client with retry configuration.
    async fn get_aws_clients(config: &Config) -> Result<(S3Client, RDSClient)> {
        let region = config
            .clone()
            .aws
            .and_then(|aws| aws.region)
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        let region_provider = Region::new(region);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
        let retry_config = RetryConfig::standard().with_max_attempts(5);
        let s3_config = S3ConfigBuilder::from(&shared_config)
            .force_path_style(force_path_style)
            .retry_config(retry_config.clone())
            .build();

        Ok((
            S3Client::from_conf(s3_config),
            RDSClient::new(&shared_config),
        ))
    }

    /// Returns initialized PostgreSQL clients for both Iris share & HNSW graph stores.
    async fn get_pgres_clients(
        config: &Config,
    ) -> Result<(IrisStore, (IrisStore, GraphPg<Aby3Store>)), Report> {
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
            log_info(format!(
                "Creating new iris store from: {:?}, schema: {}",
                db_config, db_schema
            ));
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadOnly)
                    .await?;

            IrisStore::new(&db_client).await
        }

        async fn get_hnsw_store_clients(
            config: &Config,
        ) -> Result<(IrisStore, GraphPg<Aby3Store>), Report> {
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
            log_info(format!(
                "Creating new graph store from: {:?}, schema: {}",
                db_config, db_schema
            ));
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadWrite)
                    .await?;

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
    imem_iris_stores: Arc<BothEyes<Aby3SharedIrisesRef>>,
    hnsw_iris_store: IrisStore,
    graph_store: Arc<GraphPg<Aby3Store>>,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    disable_persistence: bool,
) -> Result<Sender<JobResult>> {
    let (tx, mut rx) = mpsc::channel::<JobResult>(0); // bounded channel with no extra buffer space
    let shutdown_handler_bg = Arc::clone(shutdown_handler);
    let imem_iris_stores_bg = Arc::clone(&imem_iris_stores);
    let graph_store_bg = Arc::clone(&graph_store);
    let _result_sender_abort = task_monitor.spawn(async move {
        while let Some(result) = rx.recv().await {
            match result {
                JobResult::BatchIndexation {
                    batch_id,
                    connect_plans,
                    last_serial_id,
                    vector_ids_to_persist,
                    done_tx,
                    ..
                } => {
                    log_info(format!("Job Results :: Received: batch-id={batch_id}"));
                    // get iris shares to persist
                    let left_store = &imem_iris_stores_bg[LEFT];
                    let right_store = &imem_iris_stores_bg[RIGHT];

                    // Parallelize fetching left and right iris data using tokio::join!
                    let (left_data, right_data) = tokio::join!(
                        left_store.get_vectors_or_empty(vector_ids_to_persist.iter()),
                        right_store.get_vectors_or_empty(vector_ids_to_persist.iter())
                    );

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

                    if !disable_persistence {
                        let mut graph_tx = graph_store.tx().await?;

                        // Persist batch of Iris's to the HNSW graph store.
                        hnsw_iris_store
                            .insert_copy_irises(
                                &mut graph_tx.tx,
                                &vector_ids_to_persist,
                                &codes_and_masks,
                            )
                            .await?;
                        connect_plans.persist(&mut graph_tx).await?;
                        log_info(format!(
                            "Job Results :: Persisted graph updates: batch-id={batch_id}"
                        ));
                        let mut db_tx = graph_tx.tx;
                        set_last_indexed_iris_id(&mut db_tx, last_serial_id).await?;
                        db_tx.commit().await?;
                        log_info(format!(
                            "Job Results :: Persisted last indexed id: batch-id={batch_id}"
                        ));
                        // Update metrics with persistence status
                        metrics::counter!("genesis_batches_persisted").increment(1);
                        metrics::gauge!("genesis_indexation_complete").set(last_serial_id);
                    } else {
                        log_info(format!(
                            "Job Results :: Persistence disabled, skipping database writes for batch-id={batch_id}"
                        ));
                    }

                    let _ = done_tx.send(());
                    // Notify background task responsible for tracking pending batches.
                    shutdown_handler_bg.decrement_batches_pending_completion();
                }
                JobResult::Modification {
                    modification_id,
                    connect_plans,
                    vector_id_to_persist,
                    done_tx,
                } => {
                    log_info(format!(
                        "Job Results :: Received: modification-id={modification_id} for serial-id={}",
                        vector_id_to_persist.serial_id()
                    ));
                    // get iris shares to persist
                    let left_store = &imem_iris_stores_bg[LEFT];
                    let right_store = &imem_iris_stores_bg[RIGHT];

                    let left_iris = left_store
                        .get_vector_or_empty(&vector_id_to_persist)
                        .await;
                    let right_iris = right_store
                        .get_vector_or_empty(&vector_id_to_persist)
                        .await;

                    if !disable_persistence {
                        let mut graph_tx = graph_store_bg.tx().await?;
                        let iris_data = StoredIrisRef {
                                        id: vector_id_to_persist.serial_id() as i64,
                                        left_code: &left_iris.code.coefs,
                                        left_mask: &left_iris.mask.coefs,
                                        right_code: &right_iris.code.coefs,
                                        right_mask: &right_iris.mask.coefs,
                                    };
                        // We should ensure that the vector_id_to_persist is matching the inserted serial id
                        hnsw_iris_store.update_iris_with_version_id(
                                Some(&mut graph_tx.tx),
                                vector_id_to_persist.version_id(),
                                &iris_data,
                            )
                            .await?;
                        connect_plans.persist(&mut graph_tx).await?;
                        log_info(format!(
                            "Job Results :: Persisted graph updates: modification-id={modification_id}"
                        ));

                        let mut db_tx = graph_tx.tx;
                        set_last_indexed_modification_id(&mut db_tx, modification_id).await?;
                        db_tx.commit().await?;
                        log_info(format!(
                            "Job Results :: Persisted last indexed modification id: modification-id={modification_id}"
                        ));
                        metrics::counter!("genesis_modifications_persisted").increment(1);
                    } else {
                        log_info(format!(
                            "Job Results :: Persistence disabled, skipping database writes for modification-id={modification_id}"
                        ));
                    }

                    let _ = done_tx.send(());
                    shutdown_handler_bg.decrement_batches_pending_completion();
                },
                JobResult::Sync => unreachable!(),
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
/// * `store` - Iris PostgreSQL store provider.
/// * `max_indexation_id` - Maximum Iris serial id to which to index.
/// * `last_indexed_id` - Last Iris serial id to have been indexed.
/// * `excluded_serial_ids` - List of serial ids to be excluded from indexation.
/// * `max_modification_id` - Maximum modification id to apply after initial indexation.
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

/// Initializes HNSW graph from data previously persisted to a store.
///
/// # Arguments
///
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `config` - Application configuration instance.
/// * `graph_store` - Graph PostgreSQL store provider.
/// * `hawk_actor` - Hawk actor managing graph access & indexation.
/// * `max_index` - Optional maximum index to load (inclusive). If None, loads all data.
///
async fn init_graph_from_stores(
    config: &Config,
    iris_store: &IrisStore,
    graph_store: Arc<GraphPg<Aby3Store>>,
    hawk_actor: &mut HawkActor,
    shutdown_handler: Arc<ShutdownHandler>,
    max_indexation_id: usize,
) -> Result<()> {
    log_info(String::from("⚓️ ANCHOR: Load the database"));

    let (mut iris_loader, graph_loader) = hawk_actor.as_iris_loader().await;

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
    log_info(format!(
        "Initialize db: Loading from DB with parallelism. iris: {}, graph: {})",
        iris_db_parallelism, graph_db_parallelism
    ));

    // -------------------------------------------------------------------
    // Get total number of irises and apply max_index limit if specified
    // -------------------------------------------------------------------
    let store_len = iris_store.count_irises().await?;
    let max_index = std::cmp::min(max_indexation_id, store_len);

    load_iris_db(
        &mut iris_loader,
        iris_store,
        max_index,
        iris_db_parallelism,
        config,
        shutdown_handler,
    )
    .await
    .expect("Failed to load DB");

    iris_loader.wait_completion().await?;

    graph_loader
        .load_graph_store(&graph_store, graph_db_parallelism)
        .await?;

    Ok(())
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

/// Helper: logs & returns an error message.
fn log_error(msg: String) -> String {
    utils::log_error("Server", msg)
}

/// Helper: logs & returns an information message.
fn log_info(msg: String) -> String {
    utils::log_info("Server", msg)
}

/// Helper: logs & returns a warning message.
fn log_warn(msg: String) -> String {
    utils::log_warn("Server", msg)
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
        bail!(
            "{}",
            log_error(String::from("Missing CPU dB config settings"))
        );
    }

    Ok(())
}

/// Validates consistency of PostGres stores.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `max_indexation_id` - Maximum Iris serial id to which to index.
/// * `last_indexed_id` - Last Iris serial id to have been indexed.
///
async fn validate_consistency_of_stores(
    config: &Config,
    iris_store: &IrisStore,
    graph_store: Arc<GraphPg<Aby3Store>>,
    max_indexation_id: IrisSerialId,
    last_indexed_id: IrisSerialId,
) -> Result<()> {
    // Bail if last indexed id exceeds max indexation id
    if last_indexed_id > max_indexation_id {
        let msg = log_error(format!(
            "Last indexed id {} exceeds max indexation id {}",
            last_indexed_id, max_indexation_id
        ));
        bail!(msg);
    }

    // Bail if current Iris store length exceeds maximum constraint - should never occur.
    let store_len = iris_store.count_irises().await?;
    if store_len > config.max_db_size {
        let msg = log_error(format!(
            "Database size {} exceeds maximum allowed {}",
            store_len, config.max_db_size
        ));
        bail!(msg);
    }
    log_info(format!("Size of the database after init: {}", store_len));

    // Bail if max indexation id exceeds max id in the database
    let max_db_id = iris_store.get_max_serial_id().await?;
    if max_indexation_id as usize > max_db_id {
        let msg = log_error(format!(
            "Max indexation id {} exceeds max database id {}",
            max_indexation_id, max_db_id
        ));
        bail!(msg);
    }

    // ensure the graph store is consistent with the last persisted_indexed_id
    let mut tx = graph_store.tx().await.unwrap();
    let last_indexed_id_in_graph_left = {
        let mut graph_left = tx.with_graph(StoreId::Left);
        graph_left.get_max_serial_id().await? as u32
    };
    let last_indexed_id_in_graph_right = {
        let mut graph_right = tx.with_graph(StoreId::Right);
        graph_right.get_max_serial_id().await? as u32
    };
    if last_indexed_id_in_graph_left != last_indexed_id
        || last_indexed_id_in_graph_right != last_indexed_id
    {
        let msg = log_error(format!(
            "Last indexed id in graph store does not match last indexed id: \
             left={} :: right={} :: expected={}",
            last_indexed_id_in_graph_left, last_indexed_id_in_graph_right, last_indexed_id
        ));
        bail!(msg);
    }

    Ok(())
}
