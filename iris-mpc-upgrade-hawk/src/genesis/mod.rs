use aws_config::retry::RetryConfig;
use aws_sdk_s3::{
    config::{Builder as S3ConfigBuilder, Region},
    Client as S3Client,
};
use eyre::{bail, eyre, Report, Result};
use futures::{stream::BoxStream, StreamExt};
use iris_mpc_common::{
    config::{CommonConfig, Config, ModeOfCompute, ModeOfDeployment},
    helpers::{
        inmemory_store::InMemoryStore, shutdown_handler::ShutdownHandler, task_monitor::TaskMonitor,
    },
    postgres::{AccessMode, PostgresClient},
    server_coordination as coordinator, IrisSerialId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{GraphStore, HawkActor, HawkArgs},
    genesis::{
        self,
        state_accessor::{fetch_iris_deletions, get_last_indexed, set_last_indexed},
        state_sync::{
            Config as GenesisConfig, SyncResult as GenesisSyncResult, SyncState as GenesisSyncState,
        },
        Batch, BatchGenerator, BatchIterator, JobResult,
    },
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::graph::graph_store::GraphPg,
};
use iris_mpc_store::{DbStoredIris, Store as IrisStore};
use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::mpsc::{self, Sender},
    time::timeout,
};

const DEFAULT_REGION: &str = "eu-north-1";

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `max_indexation_id` - Maximum id to which to index iris codes.
///
pub async fn exec_main(config: Config, max_indexation_id: IrisSerialId) -> Result<()> {
    // Process: bail if config is invalid.
    validate_config(&config);

    // Process: set shutdown handler.
    let shutdown_handler = init_shutdown_handler(&config).await;

    // Coordinator: set task monitor.
    let mut background_tasks = coordinator::init_task_monitor();

    // Process: set service clients.
    let (aws_s3_client, iris_store, graph_store) = get_service_clients(&config).await?;

    let last_indexed_id = get_last_indexed(&iris_store).await?;
    let excluded_serial_ids = fetch_iris_deletions(&config, &aws_s3_client).await?;

    // Bail if stores are inconsistent.
    validate_consistency_of_stores(&config, &iris_store, max_indexation_id, last_indexed_id)
        .await?;
    // Await coordination server to start.
    let my_state = get_sync_state(
        &config,
        &iris_store,
        max_indexation_id,
        last_indexed_id,
        &excluded_serial_ids,
    )
    .await?;

    // Coordinator: await server to start.
    let is_ready_flag = coordinator::start_coordination_server(
        &config,
        &mut background_tasks,
        &shutdown_handler,
        &my_state,
    )
    .await;
    background_tasks.check_tasks();

    // Coordinator: await network state = unready.
    coordinator::wait_for_others_unready(&config).await?;

    // Coordinator: await network state = healthy.
    coordinator::init_heartbeat_task(&config, &mut background_tasks, &shutdown_handler).await?;
    background_tasks.check_tasks();

    // Coordinator: await network state = synchronized.
    let sync_result = get_sync_result(&config, &my_state).await?;
    sync_result.check_common_config()?;
    sync_result.check_genesis_config()?;

    // TODO: What should happen here - see Bryan.
    // sync_dbs_genesis(&config, &sync_result, &iris_store).await?;

    // Coordinator: escape on shutdown.
    if shutdown_handler.is_shutting_down() {
        log_warn(String::from("Shutting down has been triggered"));
        return Ok(());
    }

    // Process: initialise HNSW graph from previously indexed.
    let mut hawk_actor = get_hawk_actor(&config).await?;
    init_graph_from_stores(&config, &iris_store, &graph_store, &mut hawk_actor).await?;
    background_tasks.check_tasks();

    // Start thread for persisting indexing results to DB.
    let tx_results =
        start_results_thread(graph_store, &mut background_tasks, &shutdown_handler).await?;

    background_tasks.check_tasks();

    // Coordinator: await network state = ready.
    coordinator::set_node_ready(is_ready_flag);
    coordinator::wait_for_others_ready(&config).await?;
    background_tasks.check_tasks();

    // Process: execute main loop.
    log_info(String::from("Executing main loop"));
    exec_main_loop(
        &config,
        last_indexed_id,
        max_indexation_id,
        excluded_serial_ids,
        &iris_store,
        background_tasks,
        &shutdown_handler,
        hawk_actor,
        tx_results,
    )
    .await?;

    Ok(())
}

/// Main inner loop that performs actual indexation.
///
/// # Arguments
///
/// * `config` - Application configuration instance.
/// * `last_indexed_id` - Last Iris serial id to have been indexed.
/// * `max_indexation_id` - Maximum Iris serial id to which to index.
/// * `excluded_serial_ids` - List of serial ids to be excluded from indexing.
/// * `iris_store` - Iris PostgreSQL store provider.
/// * `task_monitor` - Tokio task monitor to coordinate with other threads.
/// * `shutdown_handler` - Handler coordinating process shutdown.
/// * `hawk_actor` - Hawk actor managing indexation & search over an HNSW graph.
/// * `tx_results` - Channel to send job results to DB persistence thread.
///
#[allow(clippy::too_many_arguments)]
async fn exec_main_loop(
    config: &Config,
    last_indexed_id: IrisSerialId,
    max_indexation_id: IrisSerialId,
    excluded_serial_ids: Vec<IrisSerialId>,
    iris_store: &IrisStore,
    mut task_monitor: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: HawkActor,
    tx_results: Sender<JobResult>,
) -> Result<()> {
    // Set Hawk handle.
    let mut hawk_handle = genesis::Handle::new(config.party_id, hawk_actor).await?;
    log_info(String::from("Hawk handle initialised"));

    // Set batch generator.
    let mut batch_generator = BatchGenerator::new(
        last_indexed_id + 1,
        max_indexation_id,
        config.max_batch_size,
        excluded_serial_ids,
    );
    log_info(String::from("Batch generator initialised"));

    // Set main loop result.
    let res: Result<()> = async {
        log_info(String::from("Entering main loop"));

        // Housekeeping.
        let now = Instant::now();
        let processing_timeout = Duration::from_secs(config.processing_timeout_secs);

        // Index until generator is exhausted.
        while let Some(Batch { data, id: batch_id }) = batch_generator.next_batch(iris_store).await? {
            // TODO: ask Bryan why this is necessary.
            let data_len = data.len();

            // Filter out any ids which have already been indexed -- there should be none
            // TODO: ask Bryan why this is necessary as generator will NEVER
            //       yield a batch with serial identifiers <= last_indexed_id.
            let data: Vec<_> = data.into_iter().filter(|db_iris| db_iris.id() > (last_indexed_id as i64)).collect();
            if data.len() < data_len {
                log_warn(format!(
                    "Filtered out previously indexed batch elements: id={} :: irises={} :: filtered={} :: time {:?}",
                    batch_id,
                    data_len,
                    data_len - data.len(),
                    now.elapsed(),
                ));
            }

            // Filter out empty batches -- this should not occur
            // TODO: ask Bryan why this is necessary as generator will NEVER
            //       yield a None or empty batch.
            if data.is_empty() {
                log_warn(format!(
                    "Skipping empty batch: id={} :: irises={} :: time {:?}",
                    batch_id,
                    data.len(),
                    now.elapsed(),
                ));
                continue;
            }

            // Collate metrics.
            metrics::histogram!("genesis_batch_duration").record(now.elapsed().as_secs_f64());

            // Coordinator: check background task processing.
            task_monitor.check_tasks();

            // TODO: ask Bryan why this is necessary as the tierator yields Batch instances
            //       and we are unecessarily destructuring and then re-instantiating.
            let batch = Batch::new(batch_id, data);
            log_info(format!(
                "Indexing new batch: {} :: time {:?}",
                batch,
                now.elapsed(),
            ));

            // Submit batch to Hawk handle for indexation.
            let result_future = hawk_handle.submit_batch(batch).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    eyre!(log_error(format!("HawkActor processing timeout: {:?}", err)))
                })??;

            // Send results to processing thread to persist to database.
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
            log_info(String::from("Main loop exited normally"));
        }
        // Error.
        Err(err) => {
            log_error(format!("HawkActor processing error: {:?}", err));

            // Clean up & shutdown.
            log_info(String::from("Initiating shutdown"));
            drop(hawk_handle);
            task_monitor.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor.check_tasks_finished();
        }
    }

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
) -> Result<(S3Client, IrisStore, GraphPg<Aby3Store>), Report> {
    /// Returns an S3 client with retry configuration.
    async fn get_aws_client(config: &Config) -> S3Client {
        // Get region from config or use default
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

        S3Client::from_conf(s3_config)
    }

    /// Returns initialized PostgreSQL clients for Iris share & HNSW graph stores.
    async fn get_pgres_clients(config: &Config) -> Result<(IrisStore, GraphPg<Aby3Store>), Report> {
        let iris_schema_name = format!(
            "{}_{}_{}",
            config.schema_name, config.environment, config.party_id
        );

        let hawk_schema_name = format!(
            "{}{}_{}_{}",
            config.schema_name, config.hnsw_schema_name_suffix, config.environment, config.party_id
        );
        // Set config.
        let db_config_iris = config
            .database
            .as_ref()
            .ok_or(eyre!("Missing database config"))?;
        let db_config_graph = config
            .cpu_database
            .as_ref()
            .ok_or(eyre!("Missing CPU database config for Hawk Genesis"))?;

        // Set postgres clients.
        let pg_client_iris = PostgresClient::new(
            &db_config_iris.url,
            iris_schema_name.as_str(),
            AccessMode::ReadOnly,
        )
        .await?;
        let pg_client_graph = PostgresClient::new(
            &db_config_graph.url,
            hawk_schema_name.as_str(),
            AccessMode::ReadWrite,
        )
        .await?;

        // Set Iris store - may take time if migrations are performed upon schemas.
        log_info(format!(
            "Creating new iris store from: {:?}",
            db_config_iris,
        ));
        let store_iris = IrisStore::new(&pg_client_iris).await?;

        // Set Graph store - may take time if migrations are performed upon schemas.
        log_info(format!(
            "Creating new graph store from: {:?}",
            db_config_graph,
        ));
        let store_graph = GraphStore::new(&pg_client_graph).await?;

        Ok((store_iris, store_graph))
    }

    let pgres_clients = get_pgres_clients(config).await?;
    let aws_s3_client = get_aws_client(config).await;

    Ok((aws_s3_client, pgres_clients.0, pgres_clients.1))
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
///
async fn get_sync_state(
    config: &Config,
    store: &IrisStore,
    max_indexation_id: IrisSerialId,
    last_indexed_id: IrisSerialId,
    excluded_serial_ids: &[IrisSerialId],
) -> Result<GenesisSyncState> {
    let db_len = store.count_irises().await? as u64;
    let common_config = CommonConfig::from(config.clone());
    let excluded_serial_ids = excluded_serial_ids.to_vec();

    let genesis_config = GenesisConfig {
        max_indexation_id,
        last_indexed_id,
        excluded_serial_ids,
    };

    Ok(GenesisSyncState {
        db_len,
        common_config,
        genesis_config,
    })
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
) -> Result<()> {
    // ANCHOR: Load the database
    log_info(String::from("⚓️ ANCHOR: Load the database"));
    let (mut iris_loader, graph_loader) = hawk_actor.as_iris_loader().await;

    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("HNSW GENESIS :: Server :: Missing database config"))?
        .load_parallelism;
    log_info(format!(
        "Initialize iris db: Loading from DB (parallelism: {})",
        parallelism
    ));

    // -------------------------------------------------------------------
    // TODO: use the number of currently processed entries for the amount
    //       to read into memory
    // -------------------------------------------------------------------
    let store_len = iris_store.count_irises().await?;
    load_db(&mut iris_loader, iris_store, store_len, parallelism)
        .await
        .expect("Failed to load DB");

    graph_loader.load_graph_store(graph_store).await?;

    Ok(())
}

/// Spawns thread responsible for persisting results from batch query processing to database.
async fn start_results_thread(
    graph_store: GraphPg<Aby3Store>,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<Sender<JobResult>> {
    let (tx, mut rx) = mpsc::channel(32); // TODO: pick some buffer value
    let shutdown_handler_bg = Arc::clone(shutdown_handler);
    let _result_sender_abort = task_monitor.spawn(async move {
        while let Some(JobResult {
            batch_id,
            identifiers,
            connect_plans,
        }) = rx.recv().await
        {
            if let (Some(first_id), Some(last_id)) = (identifiers.first(), identifiers.last()) {
                let mut graph_tx = graph_store.tx().await?;
                connect_plans.persist(&mut graph_tx).await?;
                let mut db_tx = graph_tx.tx;
                set_last_indexed(&mut db_tx, &last_id.serial_id()).await?;
                db_tx.commit().await?;

                log_info(format!(
                    "Persisted results to database for batch {}; serial ids between {} and {}",
                    batch_id,
                    first_id.serial_id(),
                    last_id.serial_id(),
                ));
            } else {
                log_warn(format!(
                    "Received empty job result for batch {} in genesis indexer results thread",
                    batch_id
                ));
            }

            shutdown_handler_bg.decrement_batches_pending_completion();
        }

        Ok(())
    });

    Ok(tx)
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

/// Loads Aurora db records from the stream into memory
///
/// # Arguments
///
/// * `actor` - Hawk actor Iris loader.
/// * `store` - Iris PostgreSQL store provider.
/// * `store_len` - Count of Iris serial identifiers.
/// * `store_load_parallelism` - Number of parallel threads to utilise when loading.
///
async fn load_db(
    actor: &mut impl InMemoryStore,
    store: &IrisStore,
    store_len: usize,
    store_load_parallelism: usize,
) -> Result<()> {
    let total_load_time = Instant::now();

    let mut all_serial_ids: HashSet<i64> = HashSet::from_iter(1..=(store_len as i64));
    actor.reserve(store_len);
    let stream_db = store
        .stream_irises_par(None, store_load_parallelism)
        .await
        .boxed();
    load_db_records(actor, &mut all_serial_ids, stream_db).await;

    if !all_serial_ids.is_empty() {
        let msg = log_error(format!(
            "Not all serial_ids were loaded: {:?}",
            all_serial_ids
        ));
        bail!(msg);
    }

    log_info(String::from("Preprocessing db"));
    actor.preprocess_db();

    log_info(format!(
        "Loaded set records from db into memory in {:?} [DB sizes: {:?}]",
        total_load_time.elapsed(),
        actor.current_db_sizes()
    ));

    eyre::Ok(())
}

/// Loads Aurora db records from the stream into memory
///
/// # Arguments
///
/// * `actor` - Hawk actor Iris loader.
/// * `all_serial_ids` - Set of Iris serial identifiers.
/// * `stream_db` - Db stream for pulling data.
///
#[allow(clippy::needless_lifetimes)]
async fn load_db_records<'a>(
    actor: &mut impl InMemoryStore,
    all_serial_ids: &mut HashSet<i64>,
    mut stream_db: BoxStream<'a, Result<DbStoredIris>>,
) {
    let mut load_summary_ts = Instant::now();
    let mut time_waiting_for_stream = Duration::from_secs(0);
    let mut time_loading_into_memory = Duration::from_secs(0);
    let mut record_counter = 0;
    while let Some(iris) = stream_db.next().await {
        // Update time waiting for the stream
        time_waiting_for_stream += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();

        let iris = iris.unwrap();

        actor.load_single_record_from_db(
            iris.serial_id() - 1,
            iris.vector_id(),
            iris.left_code(),
            iris.left_mask(),
            iris.right_code(),
            iris.right_mask(),
        );

        // Only increment db size if record has not been loaded via s3 before
        if all_serial_ids.contains(&(iris.serial_id() as i64)) {
            actor.increment_db_size(iris.serial_id() - 1);
            all_serial_ids.remove(&(iris.serial_id() as i64));
            record_counter += 1;
        }

        // Update time spent loading into memory
        time_loading_into_memory += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();
    }

    log_info(format!(
        "Aurora Loading summary => Loaded {:?} items. Waited for stream: {:?}, Loaded into memory: {:?}",
        record_counter,
        time_waiting_for_stream,
        time_loading_into_memory,
    ));
}

/// Helper: logs & returns an error message.
fn log_error(msg: String) -> String {
    genesis::log_error("Server", msg)
}

/// Helper: logs & returns an information message.
fn log_info(msg: String) -> String {
    genesis::log_info("Server", msg)
}

/// Helper: logs & returns a warning message.
fn log_warn(msg: String) -> String {
    genesis::log_warn("Server", msg)
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
fn validate_config(config: &Config) {
    // Validate modes of compute/deployment.
    if config.mode_of_compute != ModeOfCompute::Cpu {
        let msg = log_error(format!(
            "Invalid config setting: mode_of_compute: actual: {:?} :: expected: ModeOfCompute::CPU",
            config.mode_of_compute
        ));
        panic!("{}", msg);
    }

    // Validate modes of compute/deployment.
    if config.mode_of_deployment != ModeOfDeployment::Standard {
        let msg = log_error(format!(
            "Invalid config setting: mode_of_deployment: actual: {:?} :: expected: ModeOfDeployment::Standard",
            config.mode_of_deployment
        ));
        panic!("{}", msg);
    }

    log_info(format!("Mode of compute: {:?}", config.mode_of_compute));
    log_info(format!(
        "Mode of deployment: {:?}",
        config.mode_of_deployment
    ));
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
