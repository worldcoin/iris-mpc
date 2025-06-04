use aws_config::retry::RetryConfig;
use aws_sdk_rds::Client as RDSClient;
use aws_sdk_s3::{
    config::{Builder as S3ConfigBuilder, Region},
    Client as S3Client,
};
use aws_sdk_sqs::Client as SQSClient;
use chrono::Utc;
use eyre::{bail, eyre, Report, Result};
use iris_mpc_common::{
    config::{CommonConfig, Config, ModeOfCompute, ModeOfDeployment},
    helpers::{
        shutdown_handler::ShutdownHandler, smpc_request::IDENTITY_DELETION_MESSAGE_TYPE,
        sync::Modification, task_monitor::TaskMonitor,
    },
    postgres::{AccessMode, PostgresClient},
    server_coordination as coordinator, IrisSerialId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, GraphStore, HawkActor, HawkArgs, StoreId},
    genesis::{
        state_accessor::{
            get_iris_deletions, get_iris_modifications, get_last_indexed_iris_id,
            get_last_indexed_modification_id, set_last_indexed_iris_id,
        },
        state_sync::{
            Config as GenesisConfig, SyncResult as GenesisSyncResult, SyncState as GenesisSyncState,
        },
        utils, BatchGenerator, BatchIterator, Handle as GenesisHawkHandle, IndexationError,
        JobResult,
    },
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::graph::graph_store::GraphPg,
};
use iris_mpc_store::{loader::load_iris_db, Store as IrisStore};
use std::{
    sync::{atomic::AtomicU64, Arc},
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc::{self, Sender},
    time::timeout,
};

const DEFAULT_REGION: &str = "eu-north-1";

/// Process input arguments typically passed from command line.
#[derive(Debug, Clone)]
pub struct ExecutionArgs {
    // Serial idenitifer of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Batch size for indexing.
    batch_size: usize,

    // Batch size error rate to be applied.
    batch_size_error_rate: usize,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    perform_snapshot: bool,
}

/// Constructor.
impl ExecutionArgs {
    pub fn new(
        max_indexation_id: IrisSerialId,
        batch_size: usize,
        batch_size_error_rate: usize,
        perform_snapshot: bool,
    ) -> Self {
        Self {
            max_indexation_id,
            batch_size,
            batch_size_error_rate,
            perform_snapshot,
        }
    }
}

/// Information associated with inner execution context.
struct ExecutionContextInfo {
    /// Process input args.
    args: ExecutionArgs,

    // Serial idenitifer of last indexed Iris.
    last_indexed_id: IrisSerialId,

    // Serial idenitifer of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Batch size for indexing.
    batch_size: usize,

    // Batch size error rate to be applied.
    batch_size_error_rate: usize,

    // Set identifiers of Iris's to be excluded from indexation.
    excluded_serial_ids: Vec<IrisSerialId>,

    // Set of modifications to be applied.
    modifications: Vec<Modification>,

    // Maximum modification id.
    max_modification_id: i64,
}

/// Constructor.
impl ExecutionContextInfo {
    fn new(
        args: &ExecutionArgs,
        last_indexed_id: IrisSerialId,
        excluded_serial_ids: Vec<IrisSerialId>,
        modifications: Vec<Modification>,
        max_modification_id: i64,
    ) -> Self {
        Self {
            args: args.clone(),
            excluded_serial_ids,
            last_indexed_id,
            max_modification_id,
            modifications,
            batch_size: args.batch_size,
            batch_size_error_rate: args.batch_size_error_rate,
            max_indexation_id: args.max_indexation_id,
        }
    }
}

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares.
///
/// # Arguments
///
/// * `args` - Process arguments.
/// * `config` - Process configuration instance.
///
pub async fn exec(args: ExecutionArgs, config: Config) -> Result<()> {
    // Phase 0: setup.
    let (ctx, shutdown_handler, task_monitor_bg, hawk_actor, aws_rds_client, graph_store) =
        exec_setup(&args, &config).await?;
    log_info(String::from("Setup complete."));

    // Phase 1: apply delta.
    if ctx.modifications.is_empty() {
        log_info(String::from("Delta skipped ... no modifications to apply."));
    } else {
        exec_delta(&ctx).await?;
        log_info(String::from("Delta complete."));
    }

    // Phase 2: indexation.
    exec_indexation(
        &config,
        &ctx,
        hawk_actor,
        graph_store,
        task_monitor_bg,
        &shutdown_handler,
    )
    .await?;
    log_info(String::from("Indexation complete."));

    // Phase 3: snapshot.
    if !args.perform_snapshot {
        log_info(String::from("Snapshot skipped ... as requested."));
    } else {
        exec_snapshot(&config, &ctx, &aws_rds_client).await?;
        log_info(String::from("Snapshot complete."));
    };

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
    HawkActor,
    RDSClient,
    GraphPg<Aby3Store>,
)> {
    // Bail if config is invalid.
    validate_config(config)?;
    log_info(format!("Mode of compute: {:?}", config.mode_of_compute));
    log_info(format!(
        "Mode of deployment: {:?}",
        config.mode_of_deployment
    ));

    // Set shutdown handler.
    let shutdown_handler = init_shutdown_handler(config).await;

    // Set background task monitor.
    let mut task_monitor_bg = coordinator::init_task_monitor();

    // Set service clients.
    let ((aws_s3_client, aws_sqs_client, aws_rds_client), (iris_store, graph_store)) =
        get_service_clients(config).await?;
    log_info(String::from("Service clients instantiated"));

    // Set serial identifier of last indexed Iris.
    let last_indexed_id = get_last_indexed_iris_id(&graph_store).await?;
    log_info(format!(
        "Identifier of last Iris to have been indexed = {}",
        last_indexed_id,
    ));

    // Bail if stores are inconsistent.
    validate_consistency_of_stores(config, &iris_store, args.max_indexation_id, last_indexed_id)
        .await?;
    log_info(String::from("Store consistency checks OK"));

    // Set Iris serial identifiers marked for deletion and thus excluded from indexation.
    let excluded_serial_ids =
        get_iris_deletions(config, &aws_s3_client, args.max_indexation_id).await?;
    log_info(format!(
        "Deletions for exclusion count = {}",
        excluded_serial_ids.len(),
    ));

    // Set modifications that have occurred since last indexation.
    let last_indexed_modification_id = get_last_indexed_modification_id(&graph_store).await?;
    log_info(format!(
        "Identifier of last modification to have been indexed = {}",
        last_indexed_modification_id,
    ));
    let (modifications, latest_modification_id) =
        get_iris_modifications(&iris_store, last_indexed_modification_id, last_indexed_id).await?;
    log_info(format!(
        "Modifications to be applied count = {}. Last modification id = {}",
        modifications.len(),
        latest_modification_id
    ));

    // Coordinator: Await coordination server to start.
    let my_state = get_sync_state(
        config,
        args.batch_size,
        args.batch_size_error_rate,
        args.max_indexation_id,
        last_indexed_id,
        &excluded_serial_ids,
        latest_modification_id,
    )
    .await?;
    log_info(String::from("Synchronization state initialised"));

    // Coordinator: await server start.
    let current_batch_id_atomic = Arc::new(AtomicU64::new(0));
    let is_ready_flag = coordinator::start_coordination_server(
        config,
        &aws_sqs_client,
        &mut task_monitor_bg,
        &shutdown_handler,
        &my_state,
        current_batch_id_atomic,
    )
    .await;
    task_monitor_bg.check_tasks();

    // Coordinator: await network state = UNREADY.
    coordinator::wait_for_others_unready(config).await?;
    log_info(String::from("Network status = UNREADY"));

    // Coordinator: await network state = HEALTHY.
    coordinator::init_heartbeat_task(config, &mut task_monitor_bg, &shutdown_handler).await?;
    task_monitor_bg.check_tasks();
    log_info(String::from("Network status = HEALTHY"));

    // TODO: What should happen here - see Bryan.
    // sync_dbs_genesis(&config, &sync_result, &iris_store).await?;

    // Coordinator: await network state = SYNCHRONIZED.
    let sync_result = get_sync_result(config, &my_state).await?;
    sync_result.check_synced_state()?;
    log_info(String::from("Synchronization checks passed"));

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        log_warn(String::from("Shutting down has been triggered"));
        bail!("Shutdown")
        // return Ok(());
    }

    // Initialise HNSW graph from previously indexed.
    let mut hawk_actor = get_hawk_actor(config).await?;
    init_graph_from_stores(
        config,
        &iris_store,
        &graph_store,
        &mut hawk_actor,
        Arc::clone(&shutdown_handler),
    )
    .await?;
    task_monitor_bg.check_tasks();
    log_info(String::from("HNSW graph initialised from store"));

    // Coordinator: await network state = ready.
    coordinator::set_node_ready(is_ready_flag);
    coordinator::wait_for_others_ready(config).await?;
    task_monitor_bg.check_tasks();
    log_info(String::from("Network status = READY"));

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        log_warn(String::from("Shutting down has been triggered"));
        bail!("Shutdown")
        // return Ok(());
    }

    Ok((
        ExecutionContextInfo::new(
            args,
            last_indexed_id,
            excluded_serial_ids.clone(),
            modifications.clone(),
            latest_modification_id,
        ),
        shutdown_handler,
        task_monitor_bg,
        hawk_actor,
        aws_rds_client,
        graph_store,
    ))
}

/// Apply modifications since last indexation.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `modifications` - Set of indexation modifications to apply.
///
async fn exec_delta(ctx: &ExecutionContextInfo) -> Result<()> {
    let ExecutionContextInfo {
        modifications,
        max_modification_id,
        ..
    } = ctx;
    log_info(format!(
        "Applying modifications: count={} :: max-id={}",
        modifications.len(),
        max_modification_id
    ));

    // TODO: implement applying modifications
    for modification in modifications {
        log_info(format!(
            "Applying modification: type={} id={}, serial_id={}",
            modification.request_type, modification.id, modification.serial_id
        ));
        if modification.request_type == IDENTITY_DELETION_MESSAGE_TYPE {
            // throw an error
            let msg = log_error(format!(
                "HawkActor does not support deletion of identities: modification: {:?}",
                modification
            ));
            bail!(msg);
        }
        // TODO: apply modification to the graph
        // TODO: set last indexed modification id
        // set_last_indexed_modification_id(&mut db_tx, _max_modification_id).await?;
    }

    Ok(())
}

/// Index Iris's from last indexation id.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `ctx` - Execution context information.
/// * `hawk_actor` - Hawk actor managing indexation & search over an HNSW graph.
/// * `task_monitor_bg` - Tokio task monitor to coordinate with process background threads.
/// * `shutdown_handler` - Handler coordinating process shutdown.
/// * `tx_results` - Channel to send job results to DB persistence thread.
///
async fn exec_indexation(
    config: &Config,
    ctx: &ExecutionContextInfo,
    hawk_actor: HawkActor,
    graph_store: GraphPg<Aby3Store>,
    mut task_monitor_bg: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    log_info(format!("Starting indexation: batch_size={}, batch_size_error_rate={}, last_indexed_id={}, max_indexation_id={}", ctx.batch_size, ctx.batch_size_error_rate, ctx.last_indexed_id, ctx.max_indexation_id));

    // Set thread for persisting indexing results to DB.
    let tx_results =
        get_results_thread(graph_store, &mut task_monitor_bg, shutdown_handler).await?;
    task_monitor_bg.check_tasks();

    // Set dynamic batch size.
    // TODO: calculate before 1st batch and at end of each subsequent batch.
    let batch_size = match ctx.batch_size {
        // If batch size is 0 then calculate dynamic batch size based on the current graph size.
        0 => {
            // Set r: configurable parameter for error rate.
            let r = ctx.batch_size_error_rate;

            // Set m: HNSW parameter for nearest neighbors.
            let m = config.hnsw_param_M as u64;

            // Set n: current graph size (last_indexed_id).
            let n = ctx.last_indexed_id as u64;

            // Apply dynamic batch size formula: floor(N/(Mr - 1) + 1)
            let batch_size = if n > 0 {
                (n as f64 / (m as f64 * r as f64 - 1.0) + 1.0).floor() as usize
            } else {
                // Graph is empty therefore use a batch size of 1
                1
            };

            log_info(format!(
                "Dynamic batch size calculated: {} (formula: N/(Mr-1)+1, where N={}, M={}, r={})",
                batch_size, n, m, r
            ));
            batch_size
        }
        _ => {
            log_info(format!("Using static batch size: {}", ctx.batch_size));
            ctx.batch_size
        }
    };

    // Set in memory Iris stores.
    let imem_iris_stores: BothEyes<_> = [
        hawk_actor.iris_store(StoreId::Left),
        hawk_actor.iris_store(StoreId::Right),
    ];

    // Set Hawk handle.
    let mut hawk_handle = GenesisHawkHandle::new(config.party_id, hawk_actor).await?;
    log_info(String::from("Hawk handle initialised"));

    // Set batch generator.
    let mut batch_generator = BatchGenerator::new(
        ctx.last_indexed_id + 1,
        ctx.max_indexation_id,
        batch_size,
        ctx.excluded_serial_ids.clone(),
    );
    log_info(format!("Batch generator instantiated: {}", batch_generator));

    // Set indexation result.
    let res: Result<()> = async {
        log_info(String::from("Entering main indexation loop"));

        // Housekeeping.
        let now = Instant::now();
        let processing_timeout = Duration::from_secs(config.processing_timeout_secs);

        // Index until generator is exhausted.
        // N.B. assumes that generator yields non-empty batches containing serial ids > last_indexed_id.
        while let Some(batch) = batch_generator.next_batch(&imem_iris_stores).await? {
            // Coordinator: escape on shutdown.
            if shutdown_handler.is_shutting_down() {
                log_warn(String::from("Shutting down has been triggered"));
                break;
            }

            // Signal.
            log_info(format!(
                "Indexing new batch: {} :: time {:?}s",
                batch,
                now.elapsed().as_secs_f64(),
            ));
            metrics::histogram!("genesis_batch_duration").record(now.elapsed().as_secs_f64());

            // Coordinator: check background task processing.
            task_monitor_bg.check_tasks();

            // Submit batch to Hawk handle for indexation.
            let result_future = hawk_handle.submit_batch(batch).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    eyre!(log_error(format!(
                        "HawkActor processing timeout: {:?}",
                        err
                    )))
                })??;

            // Send results to processing thread responsible for persisting to database.
            tx_results.send(result).await?;
            shutdown_handler.increment_batches_pending_completion();
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
        }
    }

    Ok(())
}

/// Takes a dB snapshot.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `ctx` - Execution context information.
/// * `aws_rds_client` - AWS RDS SDK client.
///
async fn exec_snapshot(
    config: &Config,
    ctx: &ExecutionContextInfo,
    aws_rds_client: &RDSClient,
) -> Result<(), IndexationError> {
    log_info(String::from("Db snapshot begins"));

    // Set snapshot ID.
    let unix_timestamp = Utc::now().timestamp();
    let snapshot_id = format!(
        "genesis-{}-{}-{}-{}",
        ctx.last_indexed_id, ctx.max_indexation_id, ctx.args.batch_size, unix_timestamp
    );

    // Set cluster ID.
    let db_config = config.cpu_database.as_ref().unwrap();
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

/// Factory function to return a configured Hawk actor that manages HNSW graph construction & search.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
async fn get_hawk_actor(config: &Config) -> Result<HawkActor> {
    let node_addresses: Vec<String> = config
        .node_hostnames
        .iter()
        .zip(config.service_ports.iter())
        .map(|(host, port)| format!("{}:{}", host, port))
        .collect();

    let hawk_args = HawkArgs {
        party_index: config.party_id,
        addresses: node_addresses.clone(),
        request_parallelism: config.hawk_request_parallelism,
        stream_parallelism: config.hawk_stream_parallelism,
        connection_parallelism: config.hawk_connection_parallelism,
        hnsw_param_ef_constr: config.hnsw_param_ef_constr,
        hnsw_param_M: config.hnsw_param_M,
        hnsw_param_ef_search: config.hnsw_param_ef_search,
        hnsw_prng_seed: config.hawk_prng_seed,
        disable_persistence: config.cpu_disable_persistence,
        match_distances_buffer_size: config.match_distances_buffer_size,
        n_buckets: config.n_buckets,
    };

    log_info(format!(
        "Initializing HawkActor with args: party_index: {}, addresses: {:?}",
        hawk_args.party_index, node_addresses
    ));

    HawkActor::from_cli(&hawk_args).await
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
        (S3Client, SQSClient, RDSClient),
        (IrisStore, GraphPg<Aby3Store>),
    ),
    Report,
> {
    /// Returns an S3 client with retry configuration.
    async fn get_aws_clients(config: &Config) -> Result<(S3Client, SQSClient, RDSClient)> {
        let region = config
            .clone()
            .aws
            .and_then(|aws| aws.region)
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        let region_provider = Region::new(region);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let force_path_style = config.environment != "prod" && config.environment != "stage";
        let retry_config = RetryConfig::standard().with_max_attempts(5);
        let s3_config = S3ConfigBuilder::from(&shared_config)
            .force_path_style(force_path_style)
            .retry_config(retry_config.clone())
            .build();

        Ok((
            S3Client::from_conf(s3_config),
            SQSClient::new(&shared_config),
            RDSClient::new(&shared_config),
        ))
    }

    /// Returns initialized PostgreSQL clients for both Iris share & HNSW graph stores.
    async fn get_pgres_clients(config: &Config) -> Result<(IrisStore, GraphPg<Aby3Store>), Report> {
        async fn get_iris_store_client(config: &Config) -> Result<IrisStore, Report> {
            let db_schema = format!(
                "{}_{}_{}",
                config.schema_name, config.environment, config.party_id
            );
            let db_config = config
                .database
                .as_ref()
                .ok_or(eyre!("Missing database config"))?;
            log_info(format!("Creating new iris store from: {:?}", db_config));
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadOnly)
                    .await?;

            IrisStore::new(&db_client).await
        }

        async fn get_graph_store_client(config: &Config) -> Result<GraphPg<Aby3Store>, Report> {
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
            log_info(format!("Creating new graph store from: {:?}", db_config));
            let db_client =
                PostgresClient::new(&db_config.url, db_schema.as_str(), AccessMode::ReadWrite)
                    .await?;

            GraphStore::new(&db_client).await
        }

        Ok((
            get_iris_store_client(config).await?,
            get_graph_store_client(config).await?,
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
    graph_store: GraphPg<Aby3Store>,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<Sender<JobResult>> {
    let (tx, mut rx) = mpsc::channel::<JobResult>(32); // TODO: pick some buffer value
    let shutdown_handler_bg = Arc::clone(shutdown_handler);
    let _result_sender_abort = task_monitor.spawn(async move {
        while let Some(JobResult {
            batch_id,
            connect_plans,
            last_serial_id,
            ..
        }) = rx.recv().await
        {
            log_info(format!("Job Results :: Received: batch-id={}", batch_id));

            let mut graph_tx = graph_store.tx().await?;
            connect_plans.persist(&mut graph_tx).await?;
            log_info(format!(
                "Job Results :: Persisted graph updates: batch-id={}",
                batch_id
            ));

            let mut db_tx = graph_tx.tx;
            set_last_indexed_iris_id(&mut db_tx, last_serial_id).await?;
            db_tx.commit().await?;
            log_info(format!(
                "Job Results :: Persisted last indexed id: batch-id={}",
                batch_id
            ));

            log_info(format!(
                "Job Results :: Persisted to dB: batch-id={}",
                batch_id
            ));

            // Notify background task responsible for tracking pending batches.
            shutdown_handler_bg.decrement_batches_pending_completion();
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
    batch_size: usize,
    batch_size_error_rate: usize,
    max_indexation_id: IrisSerialId,
    last_indexed_id: IrisSerialId,
    excluded_serial_ids: &[IrisSerialId],
    max_modification_id: i64,
) -> Result<GenesisSyncState> {
    let common_config = CommonConfig::from(config.clone());
    let genesis_config = GenesisConfig::new(
        batch_size,
        batch_size_error_rate,
        excluded_serial_ids.to_vec(),
        last_indexed_id,
        max_indexation_id,
        max_modification_id,
    );

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
    all_states.extend(coordinator::get_others_sync_state(config).await?);
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
///
async fn init_graph_from_stores(
    config: &Config,
    iris_store: &IrisStore,
    graph_store: &GraphPg<Aby3Store>,
    hawk_actor: &mut HawkActor,
    shutdown_handler: Arc<ShutdownHandler>,
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
    // TODO: use the number of currently processed entries for the amount
    //       to read into memory
    // -------------------------------------------------------------------
    let store_len = iris_store.count_irises().await?;
    load_iris_db(
        &mut iris_loader,
        iris_store,
        store_len,
        iris_db_parallelism,
        config,
        shutdown_handler,
    )
    .await
    .expect("Failed to load DB");

    graph_loader
        .load_graph_store(graph_store, graph_db_parallelism)
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
    shutdown_handler.wait_for_shutdown_signal().await;

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

/// TODO : implement db sync genesis
#[allow(dead_code)]
async fn sync_dbs_genesis(
    _config: &Config,
    _sync_result: &GenesisSyncResult,
    _iris_store: &IrisStore,
) -> Result<()> {
    todo!("If network state decoheres then re-synchronize");
}

/// Validates application config.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
///
fn validate_config(config: &Config) -> Result<()> {
    // Validate modes of compute/deployment.
    if config.mode_of_compute != ModeOfCompute::Cpu {
        let msg = log_error(format!(
            "Invalid config setting: mode_of_compute: actual: {:?} :: expected: ModeOfCompute::CPU",
            config.mode_of_compute
        ));
        bail!("{}", msg);
    }

    // Validate modes of compute/deployment.
    if config.mode_of_deployment != ModeOfDeployment::Standard {
        let msg = log_error(format!(
            "Invalid config setting: mode_of_deployment: actual: {:?} :: expected: ModeOfDeployment::Standard",
            config.mode_of_deployment
        ));
        bail!("{}", msg);
    }

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

    Ok(())
}
